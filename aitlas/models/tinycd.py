"""TinyCD: A (Not So) Deep Learning Model For Change Detection"""

import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor
from typing import List, Optional

# Assuming BaseChangeDetection is available in your environment
from ..base import BaseChangeDetection


# -----------------------------------------------------------------------------
# Layers (from models/layers.py)
# -----------------------------------------------------------------------------

class PixelwiseLinear(nn.Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: nn.Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    nn.PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)


class MixingBlock(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        # x, y: (B, C, H, W)
        mixed = torch.stack((x, y), dim=2) # (B, C, 2, H, W)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3])) # (B, 2*C, H, W)

        # Mixing:
        return self._convmix(mixed)


class MixingMaskAttentionBlock(nn.Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: List[int],
        fout: List[int],
        generate_masked: bool = False,
    ):
        super().__init__()
        self._mixing = MixingBlock(ch_in, ch_out)
        self._linear = PixelwiseLinear(fin, fout)
        self._final_normalization = nn.InstanceNorm2d(ch_out) if generate_masked else None
        self._mixing_out = MixingBlock(ch_in, ch_out) if generate_masked else None

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)

        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class UpMask(nn.Module):
    def __init__(
        self,
        scale_factor: float,
        nin: int,
        nout: int,
    ):
        super().__init__()
        self._upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution = nn.Sequential(
            nn.Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(nin),
            nn.Conv2d(nin, nout, kernel_size=1, stride=1),
            nn.PReLU(),
            nn.InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


# -----------------------------------------------------------------------------
# Main Model Architecture (from models/change_classifier.py)
# -----------------------------------------------------------------------------

def _get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone, in_channels=3
) -> nn.ModuleList:
    # The whole model:
    entire_model = getattr(models, bkbn_name)(pretrained=pretrained).features

    # Patch first layer if input channels != 3
    if in_channels != 3:
        # For EfficientNet, the first layer in .features is usually a Conv2dNormActivation
        # entire_model[0] is the block, entire_model[0][0] is the Conv2d
        old_conv = entire_model[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )
        # Initialize weights (kaiming normal is a safe default for Relu-like activations)
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        entire_model[0][0] = new_conv

    # Slicing it:
    derived_model = nn.ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


class TinyCDModel(nn.Module):
    """
    Implementation of TinyCD: A (Not So) Deep Learning Model For Change Detection.
    Based on: https://github.com/AndreaCodegoni/Tiny_model_4_CD
    Original paper: https://link.springer.com/article/10.1007/s00521-022-08122-3
    DOI: 10.1007/s00521-022-08122-3
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone, in_channels
        )

        # Initialize mixing blocks:
        # Channel dimensions here are specific to EfficientNet-B4 at layer "3"
        # and the TinyCD architecture design.
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = nn.ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = nn.ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )

        # Final classification layer:
        # Original: self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())
        # Adaptation for AiTLAS:
        # 1. Allow variable num_classes.
        # 2. If num_classes == 1, use Sigmoid (Binary CD with BCELoss).
        # 3. If num_classes > 1, use Identity (Multiclass CD with CrossEntropyLoss), or specific config.
        
        last_act = nn.Sigmoid() if num_classes == 1 else None
        
        # NOTE: The original output dimension logic was specifically 1 channel.
        # We adjust the final layer to output `num_classes`.
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, num_classes], last_act)

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features = self._encode(ref, test)
        latents = self._decode(features)
        return self._classify(latents)

    def _encode(self, ref, test) -> List[Tensor]:
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------

class TinyCD(BaseChangeDetection):
    """
    Wrapper for TinyCD
    """
    def __init__(self, config):
        super().__init__(config)

        self.model = TinyCDModel(
            in_channels=3, #self.config.in_channels,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            bkbn_name="efficientnet_b4",
            output_layer_bkbn="3",
            freeze_backbone=False
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)