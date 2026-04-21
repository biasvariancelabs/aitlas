# Copyright contributors to the Terratorch project

import torch.nn as nn
import torch.nn.functional as F
from aitlas.models.registries import HEAD_REGISTRY


@HEAD_REGISTRY.register("SegmentationHead")
class SegmentationHead(nn.Module):
    """Segmentation head"""

    def __init__(
        self, in_channels: int, num_classes: int, channel_list: list[int] | None = None, dropout: float = 0, upsample: int = 1
    ) -> None:
        """Constructor

        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
            channel_list (list[int] | None, optional):  List with number of channels for each Conv
                layer to be created. Defaults to None.
            dropout (float, optional): Dropout value to apply. Defaults to 0.
            upsample (int, optional): Upsampling factor to apply at the end of the head. Defaults to 1.
        """
        
        super().__init__()
        self.num_classes = num_classes
        self.upsample = upsample
        if channel_list is None:
            pre_head = nn.Identity()
        else:

            def block(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1), nn.ReLU()
                )

            channel_list = [in_channels, *channel_list]
            pre_head = nn.Sequential(
                *[block(channel_list[i], channel_list[i + 1]) for i in range(len(channel_list) - 1)]
            )
            in_channels = channel_list[-1]
        dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.head = nn.Sequential(
            pre_head,
            dropout,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        # Handle list input (standardized output from backbones)
        if isinstance(x, (list, tuple)):
            x = x[0]

        out = self.head(x)

        # Upsample if needed (e.g. if the head is used with a backbone that outputs a lower resolution than the input image)
        if self.upsample > 1:
                    out = F.interpolate(out, scale_factor=self.upsample, mode='bilinear', align_corners=False)

        return out