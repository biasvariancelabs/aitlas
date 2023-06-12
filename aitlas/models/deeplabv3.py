import torch.nn as nn
from torchvision import models

from ..base import BaseSegmentationClassifier


class DeepLabV3(BaseSegmentationClassifier):
    """DeepLabV3 model implementation based on <https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.deeplabv3_resnet101>
    """

    def __init__(self, config):
        super().__init__(config)

        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=self.config.pretrained, progress=True
        )

        # change last layer to work with different number of classes
        self.model.classifier[4] = nn.Conv2d(256, self.config.num_classes, 1)

    def forward(self, x):
        return self.model(x)
