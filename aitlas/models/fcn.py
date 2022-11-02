import torch.nn as nn
from torchvision import models

from ..base import BaseSegmentationClassifier


class FCN(BaseSegmentationClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model = models.segmentation.fcn_resnet101(
            pretrained=self.config.pretrained, progress=True
        )

        # change last layer to work with different number of classes
        self.model.classifier[4] = nn.Conv2d(512, self.config.num_classes, 1)

    def forward(self, x):
        return self.model(x)
