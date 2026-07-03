from itertools import chain

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics.functional import accuracy
from torchvision import models


class MoCoV2Module(LightningModule):
    def __init__(self, base_encoder, emb_dim, num_negatives, emb_spaces=1, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        template_model = getattr(models, base_encoder)
        self.encoder_q = template_model(num_classes=self.hparams.emb_dim)
        self.encoder_k = template_model(num_classes=self.hparams.emb_dim)

        # remove fc layer
        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-1], nn.Flatten())
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-1], nn.Flatten())

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the projection heads
        self.mlp_dim = 512 * (1 if base_encoder in ["resnet18", "resnet34"] else 4)
        self.heads_q = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.mlp_dim, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, emb_dim),
                )
                for _ in range(emb_spaces)
            ]
        )
        self.heads_k = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.mlp_dim, self.mlp_dim),
                    nn.ReLU(),
                    nn.Linear(self.mlp_dim, emb_dim),
                )
                for _ in range(emb_spaces)
            ]
        )

        for param_q, param_k in zip(self.heads_q.parameters(), self.heads_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_spaces, emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(emb_spaces, 1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)
        for param_q, param_k in zip(self.heads_q.parameters(), self.heads_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_idx):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[queue_idx])
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[queue_idx, :, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[queue_idx] = ptr

    def forward(self, img_q, img_k):
        """
        Input:
            img_q: a batch of query images
            img_k: a batch of key images
        Output:
            logits, targets
        """

        # update the key encoder
        self._momentum_update_key_encoder()

        # compute query features
        v_q = self.encoder_q(img_q)

        # compute key features
        v_k = []
        for i in range(self.hparams.emb_spaces):
            # shuffle for making use of BN
            if self.use_ddp or self.use_ddp2:
                img_k[i], idx_unshuffle = batch_shuffle_ddp(img_k[i])

            with torch.no_grad():  # no gradient to keys
                v_k.append(self.encoder_k(img_k[i]))

            # undo shuffle
            if self.use_ddp or self.use_ddp2:
                v_k[i] = batch_unshuffle_ddp(v_k[i], idx_unshuffle)

        logits = []
        for i in range(self.hparams.emb_spaces):
            # compute query projections
            z_q = self.heads_q[i](v_q)  # queries: NxC
            z_q = nn.functional.normalize(z_q, dim=1)

            # compute key projections
            z_k = []
            for j in range(self.hparams.emb_spaces):
                with torch.no_grad():  # no gradient to keys
                    z_k.append(self.heads_k[i](v_k[j]))  # keys: NxC
                    z_k[j] = nn.functional.normalize(z_k[j], dim=1)

            # select positive and negative pairs
            z_pos = z_k[i]
            z_neg = self.queue[i].clone().detach()
            if i > 0:  # embedding space 0 is invariant to all augmentations
                z_neg = torch.cat(
                    [
                        z_neg,
                        *[z_k[j].T for j in range(self.hparams.emb_spaces) if j != i],
                    ],
                    dim=1,
                )

            # compute logits
            # Einstein sum is more intuitive
            l_pos = torch.einsum("nc,nc->n", z_q, z_pos).unsqueeze(-1)  # positive logits: Nx1
            l_neg = torch.einsum("nc,ck->nk", z_q, z_neg)  # negative logits: NxK

            l = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
            l /= self.hparams.softmax_temperature  # apply temperature
            logits.append(l)

            # dequeue and enqueue
            self._dequeue_and_enqueue(z_k[i], queue_idx=i)

        # targets: positive key indicators
        targets = torch.zeros(logits[0].shape[0], dtype=torch.long)
        targets = targets.type_as(logits[0])

        return logits, targets

    def training_step(self, batch, batch_idx):
        img_q, img_k = batch
        if self.hparams.emb_spaces == 1 and isinstance(img_k, torch.Tensor):
            img_k = [img_k]

        output, target = self(img_q, img_k)

        losses = []
        accuracies = []
        for out in output:
            losses.append(F.cross_entropy(out.float(), target.long()))
            accuracies = [
                accuracy(out, target, top_k=1, task="multiclass", num_classes=out.size(1))
                for out in output
            ]
        loss = torch.sum(torch.stack(losses))

        log = {"train_loss": loss}
        for i, acc in enumerate(accuracies):
            log[f"train_acc/subspace{i}"] = acc

        self.log_dict(log, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = chain(self.encoder_q.parameters(), self.heads_q.parameters())
        optimizer = optim.SGD(
            params,
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_ddp(x):  # pragma: no-cover
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):  # pragma: no-cover
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


def seco_resnet18_model(**kwargs):
    model = MoCoV2Module(base_encoder="resnet18", emb_dim=128, num_negatives=16384, emb_spaces=3)
    return model


def seco_resnet50_model(**kwargs):
    model = MoCoV2Module(base_encoder="resnet50", emb_dim=128, num_negatives=16384, emb_spaces=3)
    return model


# set recommended archs
seco_resnet18 = seco_resnet18_model
seco_resnet50 = seco_resnet50_model
