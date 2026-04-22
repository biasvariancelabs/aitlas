"""RetinaNet model for object detection"""

import math

import torch
import torch.nn as nn
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

from ..base import BaseObjectDetection


class RetinaNet(BaseObjectDetection):
    """RetinaNet model implementation

    .. note:: Based on https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2

    """

    def __init__(self, config):
        super().__init__(config)

        # Load an object detection model pre-trained on COCO
        self.model = retinanet_resnet50_fpn_v2(
            weights=(
                RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
                if self.config.pretrained
                else None
            )
        )

        # Access the internal classification head
        cls_head = self.model.head.classification_head

        # Check if we need to replace the head
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
            # Apply specific initialization (RetinaNet uses focal loss, which requires the bias to be initialized to a specific value to prevent loss instability at the start)
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
