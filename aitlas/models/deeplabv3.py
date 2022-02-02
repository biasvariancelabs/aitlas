import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from ..base import BaseSegmentationClassifier


class DeepLabV3(BaseSegmentationClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=self.config.pretrained, progress=True
        )

        # change last layer to work with different number of classes
        self.model.classifier[4] = nn.Conv2d(256, self.config.num_classes, 1)

    def forward(self, x):
        return self.model(x)


class DeepLabV3_8bands(BaseSegmentationClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=self.config.pretrained, progress=True
        )

        # Change first layer to have more than 3 channels
        #                              Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.backbone.conv1 = nn.Conv2d(8, 64, (7, 7), (2, 2), 3, bias=False)

        # change last layer to work with different number of classes
        #                             Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        self.model.classifier[4] = nn.Conv2d(256, self.config.num_classes, (1, 1))

    def forward(self, x):
        return self.model(x)
