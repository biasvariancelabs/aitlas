"""SSDLite model for object detection"""
import torch
import torch.nn as nn
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from ..base import BaseObjectDetection

# Helper function to reconstruct the prediction block 
def create_prediction_block(in_channels, out_channels, kernel_size, norm_layer):
    return nn.Sequential(
        # 3x3 depthwise with stride 1 and padding 1
        Conv2dNormActivation(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU6,
        ),
        # 1x1 projection to output channels
        nn.Conv2d(in_channels, out_channels, 1),
    )

class SSDLite(BaseObjectDetection):
    """SSDLite model implementation

    .. note:: Based on https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large

    """

    def __init__(self, config):
        super().__init__(config)

        # Load an object detection model pre-trained on COCO
        self.model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
            if self.config.pretrained
            else None
        )

        # Access the classification head
        cls_head = self.model.head.classification_head

        # Check if we need to replace the head
        if cls_head.num_columns != self.config.num_classes:
            
            # Create a new list of predictors
            new_cls_layers = nn.ModuleList()
            
            # Define the normalization layer used in SSDLite (BatchNorm with specific params)
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

            # Iterate over existing layers to create replacements
            for layer in cls_head.module_list:
                # Calculate number of anchors from the existing layer (output_channels = num_anchors * old_num_classes)
                old_out_channels = layer[-1].out_channels
                existing_num_anchors = old_out_channels // cls_head.num_columns
                # Get input channels from the first layer of the block
                in_channels = layer[0][0].in_channels
                # Create the new prediction block (output channels = num_anchors * new_num_classes)
                new_layer = create_prediction_block(
                    in_channels, 
                    existing_num_anchors * self.config.num_classes, 
                    kernel_size=3, 
                    norm_layer=norm_layer
                )
                # Apply initialization (Normal)
                for module in new_layer.modules():
                    if isinstance(module, nn.Conv2d):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.03)
                        if module.bias is not None:
                            torch.nn.init.constant_(module.bias, 0.0)
                # Append to the new layers list
                new_cls_layers.append(new_layer)

            # Replace the internal module list and update the class count attribute
            cls_head.module_list = new_cls_layers
            cls_head.num_columns = self.config.num_classes

    def forward(self, inputs, targets=None):
        return self.model.forward(inputs, targets)