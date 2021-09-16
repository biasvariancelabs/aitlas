import timm
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..base import BaseSegmentationClassifier


NHIGH = 120
NLOW = 60


class HRNetModule(nn.Module):
    def __init__(
        self, head: nn.Module, pretrained: bool = True, higher_res: bool = False
    ):
        """ Pretrained backbone for HRNet.

        Args:
            head: Output head
            pretrained: If True, uses imagenet pretrained weights
            higher_res: If True, retains higher resolution features
        """
        super().__init__()
        self.head = head
        self.backbone = timm.create_model("hrnet_w48", pretrained=pretrained)
        if higher_res:
            self.backbone.conv2.stride = (1, 1)

    def forward(self, x):
        inshape = x.shape[-2:]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)

        x = self.backbone.layer1(x)

        xl = [t(x) for i, t in enumerate(self.backbone.transition1)]
        yl = self.backbone.stage2(xl)

        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.backbone.transition2)
        ]
        yl = self.backbone.stage3(xl)

        xl = [
            t(yl[-1]) if not isinstance(t, nn.Identity) else yl[i]
            for i, t in enumerate(self.backbone.transition3)
        ]
        yl = self.backbone.stage4(xl)

        return {
            "out": F.interpolate(
                self.head(x, yl), size=inshape, mode="bilinear", align_corners=False
            )
        }


class HRNetSegHead(nn.Module):
    def __init__(self, nclasses: int = 3, higher_res: bool = False):
        """ Segmentation head for HRNet. Does not have pretrained weights.

        Args:
            nclasses: Number of output classes
            higher_res: If True, retains higher resolution features
        """
        super().__init__()
        self.res_modifier = 2 if higher_res else 1
        self.projection = nn.Sequential(
            nn.Conv2d(976, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, nclasses, 1),
        )

    def forward(self, x, yl):
        mod = self.res_modifier
        low_level = torch.cat(
            [F.interpolate(feat, (NHIGH * mod, NHIGH * mod)) for feat in [x, *yl]], 1
        )
        out = self.projection(low_level)
        return out


class HRNet(BaseSegmentationClassifier):
    def __init__(self, config, higher_res=False):
        super().__init__(config)
        self.model = HRNetModule(
            HRNetSegHead(self.config.num_classes, higher_res),
            self.config.pretrained,
            higher_res,
        )

    def forward(self, x):
        return self.model(x)
