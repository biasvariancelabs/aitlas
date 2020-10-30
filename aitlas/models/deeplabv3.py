import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from ..base import BaseSegmentationClassifier


class DeepLabV3(BaseSegmentationClassifier):

    def __init__(self, config):
        BaseSegmentationClassifier.__init__(self, config)

        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=self.config.pretrained, progress=True
        )

        # change last layer to work with different number of classes
        self.model.classifier[4] = nn.Conv2d(256, self.config.num_classes, 1)

    def forward(self, x):
        return self.model.forward(x)

