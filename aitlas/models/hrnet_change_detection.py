"""
HRNet model for change detection
"""

import timm
import torch
from torch import nn
from torch.nn import functional as F

from ..base import BaseChangeDetection


class SiameseHRNetModule(nn.Module):
    """Siamese HRNet module for change detection."""

    def __init__(
        self,
        head: nn.Module,
        pretrained: bool = True,
        higher_res: bool = False,
        backbone_name: str = "hrnet_w18",
    ):
        super().__init__()
        self.head = head
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
        self.backbone_name = backbone_name

        # Optional higher resolution: reduce downsampling in early conv
        if higher_res:
            if hasattr(self.backbone, "conv2"):
                self.backbone.conv2.stride = (1, 1)

    def forward_encoder(self, x):
        # Standard HRNet stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)

        # Layer1
        x_layer1 = self.backbone.layer1(x)

        # Stage 2
        xl = [t(x_layer1) for t in self.backbone.transition1]
        yl = self.backbone.stage2(xl)

        # Stage 3
        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.backbone.transition2)
        ]
        yl = self.backbone.stage3(xl)

        # Stage 4
        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.backbone.transition3)
        ]
        yl = self.backbone.stage4(xl)

        # x_layer1: low-res, yl: list of multi-res features (high→low)
        return x_layer1, yl

    def forward(self, x1, x2):
        inshape = x1.shape[-2:]

        # Siamese encoding
        feat1_layer1, feat1_yl = self.forward_encoder(x1)
        feat2_layer1, feat2_yl = self.forward_encoder(x2)

        out = self.head(feat1_layer1, feat1_yl, feat2_layer1, feat2_yl)

        return F.interpolate(
            out,
            size=inshape,
            mode="bilinear",
            align_corners=False,
        )


class HRNetCDHead(nn.Module):
    """
    Change detection head for HRNet
    """

    def __init__(
        self,
        nclasses: int = 2,
        higher_res: bool = False,
        backbone_name: str = "hrnet_w18",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.higher_res = higher_res

        # Channel definitions per HRNet variant
        # layer1: 256 channels for all
        if backbone_name == "hrnet_w18":
            feat_channels = [256, 18, 36, 72, 144]
        elif backbone_name == "hrnet_w32":
            feat_channels = [256, 32, 64, 128, 256]
        elif backbone_name == "hrnet_w48":
            feat_channels = [256, 48, 96, 192, 384]
        else:
            raise ValueError(f"Unsupported HRNet backbone: {backbone_name}")

        self.num_feats = len(feat_channels)  # layer1 + 4 branches

        # Per-feature projection to a common channel dimension
        fused_channels = 64
        self.proj_per_feat = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, fused_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fused_channels),
                    nn.ReLU(inplace=True),
                )
                for c in feat_channels
            ]
        )

        # Siamese fusion: concat(f1, f2)
        siamese_in_channels = fused_channels * 2

        self.projection = nn.Sequential(
            nn.Conv2d(siamese_in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, nclasses, kernel_size=1),
        )

    def _fuse_single_image(self, x_layer1, yl):
        """
        Fuse multi-scale features for a single image:
        - x_layer1: low-res feature (256 channels)
        - yl: list of 4 features from stage4 branches (high→low resolution)
        """
        feats = [x_layer1, *yl]

        # Use the highest-resolution branch from stage4 as fusion target
        target_h, target_w = yl[0].shape[-2:]

        fused = None
        for feat, proj in zip(feats, self.proj_per_feat):
            feat_up = F.interpolate(
                feat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            feat_proj = proj(feat_up)
            fused = feat_proj if fused is None else fused + feat_proj

        return fused

    def forward(self, x1, yl1, x2, yl2):
        # Fuse features per image
        fused1 = self._fuse_single_image(x1, yl1)
        fused2 = self._fuse_single_image(x2, yl2)

        # Siamese fusion: concat(f1, f2)
        combined = torch.cat([fused1, fused2], dim=1)

        out = self.projection(combined)
        return out


class HRNetChangeDetection(BaseChangeDetection):
    """HRNet model for change detection."""

    def __init__(self, config):
        super().__init__(config)

        higher_res = getattr(self.config, "higher_res", False)
        backbone_name = getattr(self.config, "backbone", "hrnet_w18")

        head = HRNetCDHead(
            nclasses=self.config.num_classes,
            higher_res=higher_res,
            backbone_name=backbone_name,
        )

        self.model = SiameseHRNetModule(
            head=head,
            pretrained=self.config.pretrained,
            higher_res=higher_res,
            backbone_name=backbone_name,
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)
