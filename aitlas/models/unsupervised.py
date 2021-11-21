import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler

from ..base import BaseMulticlassClassifier
from ..clustering import Kmeans, cluster_assign
from .schemas import UnsupervisedDeepMulticlassClassifierSchema


class UnsupervisedDeepMulticlassClassifier(BaseMulticlassClassifier):

    schema = UnsupervisedDeepMulticlassClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.learning_rate = self.config.learning_rate
        self.weight_decay = self.config.weight_decay
        self.number_of_clusters = self.config.number_of_clusters
        self.sobel = self.config.sobel

        self.model = vgg16(sobel=self.sobel)
        self.fd = int(self.model.top_layer.weight.size()[1])
        self.model.top_layer = None

        self.deepcluster = Kmeans(self.number_of_clusters)

        self.reassign = 1

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        """Overriding train epoch to implement the custom logic for the unsupervised classifier"""
        self.model.top_layer = None
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1]
        )

        # get the original dataset
        dataset = dataloader.dataset

        # get the features for the whole dataset
        features = compute_features(
            dataloader,
            self.model,
            len(dataloader),
            dataset.config.batch_size,
            self.device,
        )

        # cluster the features]
        clustering_loss = self.deepcluster.cluster(features)

        # assign pseudo-labels
        train_dataset = cluster_assign(self.deepcluster.images_lists, dataset)

        # uniformly sample per target
        sampler = UnifLabelSampler(
            int(self.reassign * len(train_dataset)), self.deepcluster.images_lists
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=dataset.batch_size,
            num_workers=dataset.num_workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(self.model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).to(self.device))
        self.model.classifier = nn.Sequential(*mlp)
        self.model.top_layer = nn.Linear(self.fd, len(self.deepcluster.images_lists))
        self.model.top_layer.weight.data.normal_(0, 0.01)
        self.model.top_layer.bias.data.zero_()
        self.model.top_layer.to(self.device)

        # create an optimizer for the last fc layer
        optimizer_tl = torch.optim.SGD(
            self.model.top_layer.parameters(),
            lr=self.learning_rate,
            weight_decay=10 ** self.weight_decay,
        )

        # send both optimizers
        optimizers = (
            optimizer,
            optimizer_tl,
        )

        return super().train_epoch(
            epoch, train_dataloader, optimizers, criterion, iterations_log
        )

    def forward(self, x):
        return self.model.forward(x)


def compute_features(dataloader, model, N, batch, device):
    """Compute features for images"""
    model.eval()

    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        input_var = torch.autograd.Variable(input_tensor.to(device), volatile=True)
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1])).astype("float32")

        if i < len(dataloader) - 1:
            features[i * batch : (i + 1) * batch] = aux.astype("float32")
        else:
            # special treatment for final batch
            features[i * batch :] = aux.astype("float32")

    return features


class VGG(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()
        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(input_dim, batch_norm):
    layers = []
    in_channels = input_dim
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(sobel=False, bn=True, out=1000):
    dim = 2 + int(not sobel)
    model = VGG(make_layers(dim, bn), out, sobel)
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel),
            )
            res[i * size_per_pseudolabel : (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[: self.N].astype("int")

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N
