from typing import List

import torch
from torch import nn as nn

    
class PatchMLP(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            scale: int = 1,
            resolution: float = 0.2,
            embed_dim: int = 768,
            patch_size: int = 10,
            bias: bool = True,
            mlp: List[int] = [],
            ):
        super().__init__()
        self.scale = scale
        self.res = int(10 / resolution)
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        layers = []
        for i in range(len(mlp) - 1):
            layers.extend(
                [
                    nn.Linear(mlp[i], mlp[i + 1]),
                    nn.LayerNorm(mlp[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp  = nn.Sequential(*layers)

    def forward(self, x):
        x = self.patch_embed(x)
        grid_size = max(self.res // self.patch_size, 1)
        x = x.unfold(2, grid_size, grid_size).unfold(3, grid_size, grid_size)
        x = x.flatten(4, 5)
        x = x.unfold(2, self.scale, self.scale).unfold(3, self.scale, self.scale)
        x = x.flatten(2, 3).permute(0, 1, 2, 4, 5, 3).flatten(3, 5)
        x = torch.permute(x,(0,2,3,1))
        x = x.flatten(0,1)
        x = self.mlp(x)
        return x

class PatchMLPMulti(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            resolution: float = 0.2,
            embed_dim: int = 768,
            patch_size: int = 10,
            bias: bool = True,
            mlp: List[int] = [],
            ):
        super().__init__()
        self.patch_size = patch_size
        self.res = int(10 / resolution)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        layers = []
        for i in range(len(mlp) - 1):
            layers.extend(
                [
                    nn.Linear(mlp[i], mlp[i + 1]),
                    nn.LayerNorm(mlp[i + 1]),
                    nn.ReLU(),
                ]
            )
        self.mlp  = nn.Sequential(*layers)

    def forward(self, x, scale):
        x = self.patch_embed(x)
        grid_size = (self.res // self.patch_size, self.res // self.patch_size)
        x = x.unfold(2, grid_size[0], grid_size[0]).unfold(3, grid_size[1], grid_size[1])
        x = x.flatten(4, 5)
        x = x.unfold(2, scale, scale).unfold(3, scale, scale)
        x = x.flatten(2, 3).permute(0, 1, 2, 4, 5, 3).flatten(3, 5)
        x = torch.permute(x,(0,2,3,1))
        x = x.flatten(0,1)
        x = self.mlp(x)
        return x