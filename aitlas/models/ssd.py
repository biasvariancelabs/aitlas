"""SSD model for object detection"""
import torch
import torch.nn as nn
from torchvision.models.detection import (
    SSD300_VGG16_Weights,
    ssd300_vgg16,
)
from ..base import BaseObjectDetection


class SSD(BaseObjectDetection):
    """SSD model implementation

    .. note:: Based on https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssd300_vgg16.html#ssd300-vgg16

    """

    def __init__(self, config):
        super().__init__(config)

        # Load an object detection model pre-trained on COCO
        self.model = ssd300_vgg16(
            weights=SSD300_VGG16_Weights.COCO_V1
            if self.config.pretrained
            else None
        )

        # Access the classification head (can be multiple layers stored in ModuleList)
        cls_head = self.model.head.classification_head

        # Check if we need to replace the head
        if cls_head.num_columns != self.config.num_classes:         
            
            # Create a new list of predictors
            new_cls_layers = nn.ModuleList()
            
            # Iterate over the existing layers to create replacements
            for layer in cls_head.module_list:
                # We need to deduce the number of anchors for this specific layer (num_anchors = output_channels // old_num_classes)
                existing_num_anchors = layer.out_channels // cls_head.num_columns
                # Create a new layer with the correct output size for the new classes
                new_layer = nn.Conv2d(
                    in_channels=layer.in_channels,
                    out_channels=existing_num_anchors * self.config.num_classes,
                    kernel_size=3,
                    padding=1
                )      
                # Apply specific initialization (Xavier Uniform)
                torch.nn.init.xavier_uniform_(new_layer.weight)
                if new_layer.bias is not None:
                    torch.nn.init.constant_(new_layer.bias, 0.0)
                # Append to the new layers list
                new_cls_layers.append(new_layer)

            # Replace the internal module list and update the class count attribute
            cls_head.module_list = new_cls_layers
            cls_head.num_columns = self.config.num_classes

    def forward(self, inputs, targets=None):
        return self.model.forward(inputs, targets)