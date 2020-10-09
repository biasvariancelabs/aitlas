import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from ..base import BaseSegmentation
from .schemas import DeepLabV3Schema


class DeepLabV3(BaseSegmentation):
    schema = DeepLabV3Schema

    def __init__(self, config):
        BaseSegmentation.__init__(self, config)

        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True
        )

        # change last layer to work with different number of classes
        self.model.classifier[4] = nn.Conv2d(256, self.config.num_classes, 1)
        # add final layer
        self.model.classifier.add_module("5", nn.Tanh())

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, *input, **kwargs):
        return self.model(*input)["out"]

    def load_criterion(self):
        return torch.nn.MSELoss(reduction="mean")

    def load_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
