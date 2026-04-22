"""FCOS model for object detection"""

import math

import torch
import torch.nn as nn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn

from ..base import BaseObjectDetection


class FCOS(BaseObjectDetection):
    """FCOS model implementation

    .. note:: Based on https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn

    """

    def __init__(self, config):
        super().__init__(config)

        # Load an object detection model pre-trained on COCO
        self.model = fcos_resnet50_fpn(
            weights=(
                FCOS_ResNet50_FPN_Weights.COCO_V1 if self.config.pretrained else None
            )
        )

        # Access the internal classification head
        cls_head = self.model.head.classification_head

        # If the number of classes differs from the pre-trained model, we must replace the classification layer
        if cls_head.num_classes != self.config.num_classes:
            # Get existing parameters
            in_channels = cls_head.cls_logits.in_channels
            num_anchors = cls_head.num_anchors
            # Create a new classification convolution layer (output channels = num_anchors * num_classes)
            new_cls_logits = nn.Conv2d(
                in_channels,
                num_anchors * self.config.num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            # Apply specific initialization (FCOS uses focal loss, which requires the bias to be initialized to a specific value to prevent loss instability at the start)
            prior_probability = 0.01
            torch.nn.init.normal_(new_cls_logits.weight, std=0.01)
            torch.nn.init.constant_(
                new_cls_logits.bias,
                -math.log((1 - prior_probability) / prior_probability),
            )
            # Replace the layer and update the attributes
            cls_head.cls_logits = new_cls_logits
            cls_head.num_classes = self.config.num_classes

    def forward(self, inputs, targets=None):
        return self.model.forward(inputs, targets)
