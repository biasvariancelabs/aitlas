import torch
from torch import nn, optim
import torchvision
from pytorch_lightning import LightningModule

MINUSINF = -100000000


class MoCoV2CACoModule(LightningModule):

    def __init__(self, base_encoder, emb_dim, num_negatives, emb_spaces=1, datamodule=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.datamodule = datamodule

        # create the encoders
        template_model = getattr(torchvision.models, base_encoder)
        self.encoder_q = template_model(num_classes=self.hparams.emb_dim)
        self.encoder_k = template_model(num_classes=self.hparams.emb_dim)

        # remove fc layer
        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-1], nn.Flatten())
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-1], nn.Flatten())

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the projection heads
        self.mlp_dim = 512 * (1 if base_encoder in ['resnet18', 'resnet34'] else 4)
        self.heads_q = nn.ModuleList([
            nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, emb_dim))
            for _ in range(emb_spaces)
        ])
        self.heads_k = nn.ModuleList([
            nn.Sequential(nn.Linear(self.mlp_dim, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, emb_dim))
            for _ in range(emb_spaces)
        ])

        for param_q, param_k in zip(self.heads_q.parameters(), self.heads_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_spaces, emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(emb_spaces, 1, dtype=torch.long))


def caco_resnet18_model(**kwargs):
    model = MoCoV2CACoModule(base_encoder='resnet18', emb_dim=128, num_negatives=16384, emb_spaces=3, datamodule=None)
    return model

def caco_resnet50_model(**kwargs):
    model = MoCoV2CACoModule(base_encoder='resnet50', emb_dim=128, num_negatives=16384, emb_spaces=3, datamodule=None)
    return model

# set recommended archs
caco_resnet18 = caco_resnet18_model
caco_resnet50 = caco_resnet50_model