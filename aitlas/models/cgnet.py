"""CGNet: Change Guiding Network for Change Detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..base import BaseChangeDetection

# -----------------------------------------------------------------------------
# Helper Modules (from network/CGNet.py)
# -----------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChangeGuideModule(nn.Module):
    """
    Change Guide Module: A self-attention mechanism that captures long-range
    dependencies and is guided by a change prior map to focus on altered regions.
    """
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        # The guiding map is interpolated to the current feature size
        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        guiding_map = torch.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out

# -----------------------------------------------------------------------------
# Main CGNet Model (from network/CGNet.py)
# -----------------------------------------------------------------------------

class CGNetModel(nn.Module):
    """
    Implementation of CGNet: Change Guiding Network
    Based on: https://github.com/ChengxiHAN/CGNet-CD
    Original paper: https://ieeexplore.ieee.org/document/10234560
    DOI: 10.1109/JSTARS.2023.3310208
    """
    def __init__(self, in_channels=3, num_classes=2, pretrained=True):
        super(CGNetModel, self).__init__()
        
        vgg16_bn = models.vgg16_bn(pretrained=pretrained)
        
        # Patch first conv layer if input channels are not standard RGB (3)
        if in_channels != 3:
            old_conv = vgg16_bn.features[0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            vgg16_bn.features[0] = new_conv

        # Backbone Slices (VGG-16 BN)
        self.inc = vgg16_bn.features[:5]      # 64 channels
        self.down1 = vgg16_bn.features[5:12]  # 128 channels
        self.down2 = vgg16_bn.features[12:22] # 256 channels
        self.down3 = vgg16_bn.features[22:32] # 512 channels
        self.down4 = vgg16_bn.features[32:42] # 512 channels

        # Fusion channel reduction
        self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

        # Decoders
        # The internal guide map decoder always outputs 1 channel as it's used for attention guidance.
        self.decoder = nn.Sequential(
            BasicConv2d(512, 64, 3, 1, 1), 
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        # The final decoder outputs num_classes as requested by the framework.
        self.decoder_final = nn.Sequential(
            BasicConv2d(128, 64, 3, 1, 1), 
            nn.Conv2d(64, num_classes, 1)
        )

        # Change Guide Modules
        self.cgm_2 = ChangeGuideModule(256)
        self.cgm_3 = ChangeGuideModule(512)
        self.cgm_4 = ChangeGuideModule(512)

        # Hierarchical Decoder Modules
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4 = BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_module3 = BasicConv2d(768, 256, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(384, 128, 3, 1, 1)

    def forward(self, A, B):
        size = A.size()[2:]
        
        # Siamese Encoder
        l1_pre_A = self.inc(A)
        l1_A = self.down1(l1_pre_A)
        l2_A = self.down2(l1_A)
        l3_A = self.down3(l2_A)
        l4_A = self.down4(l3_A)

        l1_pre_B = self.inc(B)
        l1_B = self.down1(l1_pre_B)
        l2_B = self.down2(l1_B)
        l3_B = self.down3(l2_B)
        l4_B = self.down4(l3_B)

        # Feature Fusion via Concatenation
        layer1 = self.conv_reduce_1(torch.cat((l1_B, l1_A), dim=1))
        layer2 = self.conv_reduce_2(torch.cat((l2_B, l2_A), dim=1))
        layer3 = self.conv_reduce_3(torch.cat((l3_B, l3_A), dim=1))
        layer4 = self.conv_reduce_4(torch.cat((l4_B, l4_A), dim=1))

        # Initial Guide Map Generation (Internal Guidance)
        layer4_up = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        change_map = self.decoder(layer4_up)

        # Hierarchical Change-Guided Decoding
        layer4 = self.cgm_4(layer4, change_map)
        f4 = self.decoder_module4(torch.cat([self.upsample2x(layer4), layer3], dim=1))
        
        layer3 = self.cgm_3(f4, change_map)
        f3 = self.decoder_module3(torch.cat([self.upsample2x(layer3), layer2], dim=1))
        
        layer2 = self.cgm_2(f3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], dim=1))

        # Final Prediction
        final_map = self.decoder_final(layer1)
        final_map_out = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return final_map_out


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------

class CGNet(BaseChangeDetection):
    """
    Wrapper for CGNet: Change Guiding Network
    """
    def __init__(self, config):
        super().__init__(config)

        self.model = CGNetModel(
            in_channels=3, # self.config.in_channels if hasattr(self.config, 'in_channels') else 3,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained
        )

    def forward(self, x1, x2):
        # Aitlas expects a single tensor output for logits.
        return self.model(x1, x2)