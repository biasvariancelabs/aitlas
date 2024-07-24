"""DeepLabV3 model"""
import torch.nn as nn
from torchvision import models

from ..base import BaseSegmentationClassifier
from .schemas import DeepLabV3ModelSchema


class DeepLabV3(BaseSegmentationClassifier):
    """DeepLabV3 model implementation

    .. note:: Based on https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.deeplabv3_resnet101

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
    
    
class DeepLabV3_13_bands(BaseSegmentationClassifier):
    """DeepLabV3 model implementation for input with more than 3 channels

    .. note:: Based on https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.deeplabv3_resnet101

    """

    schema = DeepLabV3ModelSchema

    def __init__(self, config):
        super().__init__(config)

        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained = self.config.pretrained, progress=True, num_classes= self.config.num_classes
        )
        
        self.model.backbone.conv1 = nn.Conv2d(self.config.input_dim, 64, self.config.num_classes, 1, bias=False) #to accept the 13 channels

        self.model.classifier[4] = nn.Conv2d(256, self.config.num_classes, 1)

    def forward(self, x):
        return self.model(x)
