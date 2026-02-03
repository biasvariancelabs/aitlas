"""Siamese UNet model for change detection (Faithful Implementation)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..base import BaseChangeDetection


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # In the library, they concatenate (in + skip) then pass to conv1
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip=None):
        # Upsample the input (x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # Concatenate with skip connection if provided
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SiameseUnet(nn.Module):
    """
    Siamese U-Net for change detection.
    Based on: https://github.com/likyoo/change_detection.pytorch implementation of U-Net for change detection.
    """
    def __init__(self, in_channels=3, num_classes=2, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # --- Encoder Definition ---
        # Initialize standard model
        self.encoder = models.resnet50(pretrained=pretrained)
        
        # Patch first conv layer for arbitrary input channels
        if in_channels != 3:
            old_conv = self.encoder.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            self.encoder.conv1 = new_conv
        
        # Remove unused layers
        del self.encoder.fc
        del self.encoder.avgpool

        # --- Dynamic Channel Inference ---
        # We create a dummy input on CPU to pass through the encoder
        # The size (64x64) doesn't matter much, as long as it passes through the layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, 64, 64)
            features = self.forward_encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in features]
        
        # encoder_channels will now automatically be:
        # [64, 256, 512, 1024, 2048] for ResNet50
        # [64, 64, 128, 256, 512]    for ResNet18 / ResNet34

        # Standard decoder filters from the library
        decoder_channels = [256, 128, 64, 32, 16]

        # --- Decoder Definition ---
        self.blocks = nn.ModuleList()
        
        # Head (Input to Block 0) is f5 * 2 (concatenated)
        head_channels = encoder_channels[-1] * 2
        
         # Dynamically build blocks using the inferred encoder_channels list
        self.blocks.append(DecoderBlock(head_channels, encoder_channels[3] * 2, decoder_channels[0]))
        self.blocks.append(DecoderBlock(decoder_channels[0], encoder_channels[2] * 2, decoder_channels[1]))
        self.blocks.append(DecoderBlock(decoder_channels[1], encoder_channels[1] * 2, decoder_channels[2]))
        self.blocks.append(DecoderBlock(decoder_channels[2], encoder_channels[0] * 2, decoder_channels[3]))
        self.blocks.append(DecoderBlock(decoder_channels[3], 0, decoder_channels[4]))

        # --- Final Segmentation Head ---
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], self.num_classes, kernel_size=3, padding=1)

    def forward_encoder(self, x):
        """Extract features from ResNet encoder stages"""
        features = []
        
        # Stage 1: Conv1 -> BN -> ReLU
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x) # f1 (H/2)
        
        # MaxPool (part of Stage 2 usually, but separated for feature list)
        x = self.encoder.maxpool(x)
        
        # Stage 2: Layer 1
        x = self.encoder.layer1(x)
        features.append(x) # f2 (H/4)
        
        # Stage 3: Layer 2
        x = self.encoder.layer2(x)
        features.append(x) # f3 (H/8)
        
        # Stage 4: Layer 3
        x = self.encoder.layer3(x)
        features.append(x) # f4 (H/16)
        
        # Stage 5: Layer 4
        x = self.encoder.layer4(x)
        features.append(x) # f5 (H/32)
        
        return features

    def forward(self, x1, x2):
        # 1. Siamese Encoding
        features1 = self.forward_encoder(x1)
        features2 = self.forward_encoder(x2)
        
        # 2. Fusion (Concatenation)
        # We reverse the features to start from the deepest (f5)
        features1 = features1[::-1]
        features2 = features2[::-1]
        
        # 3. Decoding
        # Initial input is the concatenated deepest features
        x = torch.cat([features1[0], features2[0]], dim=1)
        
        # Iterate through blocks.
        # Skips start from the second deepest (f4) down to f1.
        # The last block (Block 4) has no skip connection in this config.
        skips_len = len(features1) - 1
        
        for i, block in enumerate(self.blocks):
            if i < skips_len:
                # Concatenate the skip connections from both branches
                skip = torch.cat([features1[i+1], features2[i+1]], dim=1)
                x = block(x, skip)
            else:
                # Final block (upsample to original size)
                x = block(x, skip=None)
                
        # 4. Final Prediction
        x = self.segmentation_head(x)
        
        return x


class UnetChangeDetection(BaseChangeDetection):
    """
    Wrapper for the Siamese U-Net for change detection.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = SiameseUnet(
            in_channels=3, #self.config.in_channels,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)