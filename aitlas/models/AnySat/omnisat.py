from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn

from .utils.pos_embed import get_2d_sincos_pos_embed_with_scale
from .utils.utils import PatchDropout, trunc_normal_
from .utils.utils_ViT import Block, CrossRPEBlock


class OmniSatModule(nn.Module):
    """
    Initialiazes OmniSat encoding module.
    Args:
        projectors (dict): dict of all possible projectors
        modalities (list): list of modalities to use
        num_patches (int): number of patches by observation. Is the same for all modalities by alignement
        embed_dim (int): embed_dim of transformer blocks
        depth (int): depth of transformer blocks
        num_heads (int):  num_heads of transformer blocks
        mlp_ratio (float): mlp_ratio of transformer blocks
        qkv_bias (bool): for transformer blocks
        qk_scale: for transformer blocks
        class_token (bool): if True, add a class token
        pre_norm (bool): False, for transformer blocks
        drop_rate (float): drop_rate for transformer blocks
        pos_drop_rate (float): pos_drop_rate for transformer blocks
        patch_drop_rate (float): patch_drop_rate for transformer blocks
        drop_path_rate (float): drop_path_rate for transformer blocks
        attn_drop_rate (float): attn_dropout_rate for transformer blocks
        norm_layer (Optional[Callable]): norm layer for transformer blocks
    """

    def __init__(
        self,
        spatial_encoder: nn.Module,
        projectors: dict = {},
        modalities: list = [],
        num_patches: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        class_token: bool = True,
        pre_norm: bool = False,
        drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Optional[Callable] = None,
        height: int = 6,
        width: int = 6,
        scale: int = 1,
        keep_subpatch: bool = False,
        modality_keep: str = "",
        flash_attn: bool = True,
    ):

        super(OmniSatModule, self).__init__()
        self.modalities = modalities

        num_patches = num_patches // (scale * scale)
        self.height = height // scale
        self.width = width // scale
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_patches = num_patches + self.num_prefix_tokens
        self.embed_dim = embed_dim
        self.keep_subpatch = keep_subpatch
        self.modality_keep = modality_keep

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.pos_embed = get_2d_sincos_pos_embed_with_scale(
            embed_dim, int(num_patches**0.5), cls_token=class_token, scale=scale
        )
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        for i in range(len(modalities)):
            if modalities[i].split("-")[-1] == "mono":
                m = "-".join(modalities[i].split("-")[:-1])
            else:
                m = modalities[i]
            setattr(self, "_".join(["projector", modalities[i]]), projectors[m])

        self.spatial_encoder = spatial_encoder

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    flash_attn=flash_attn,
                )
                for i in range(depth)
            ]
            + [
                CrossRPEBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    modalities=modalities,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[-1],
                    norm_layer=norm_layer,
                    num_patches=self.num_patches,
                    height=self.height,
                    width=self.width,
                    scale=scale,
                    modis=("modis" in modalities),
                )
            ]
        )
        trunc_normal_(self.cls_token, std=0.02)

    def forward_proj(self, x):
        """
        Forward function until masking used during pretraining
        """
        tokens = []
        out = {}
        self.pos_embed = self.pos_embed.to(x[self.modalities[0]].device)
        for modality in self.modalities:
            if modality in ["aerial", "spot", "aerial-flair", "naip", "planet"]:
                token = getattr(self, "_".join(["projector", modality]))(x[modality])
            elif modality.split("-")[-1] == "mono":
                token = getattr(self, "_".join(["projector", modality]))(
                    x[modality].unsqueeze(1),
                    torch.zeros(x[modality].shape[0], 1).to(x[modality].device) + 120,
                )
                token = token.view(token.shape[0], token.shape[1], -1).permute(0, 2, 1)
            else:
                if "_".join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, "_".join(["projector", modality]))(
                        x[modality],
                        x["_".join([modality, "dates"])],
                        x["_".join([modality, "mask"])],
                    )
                else:
                    token = getattr(self, "_".join(["projector", modality]))(
                        x[modality], x["_".join([modality, "dates"])]
                    )
            token = self.spatial_encoder(token, modality)
            token = token.view(-1, self.height * self.width, self.embed_dim)
            out["_".join(["tokens", modality])] = token
            tokens.append(
                out["_".join(["tokens", modality])] + self.pos_embed[:, 1:, :]
            )
        tokens = torch.cat(tokens, dim=1)
        return tokens, out

    def forward_transformer(self, x, mask):
        """
        Forward function after masking used during pretraining
        """
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(
                x.shape[0], -1, -1
            )
            tokens = torch.cat((cls_tokens, x), dim=1)
        else:
            tokens = x

        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, mask)
        return tokens

    def forward(self, x):
        """
        Complete forward function during training
        """
        tokens = []
        out = {}
        self.pos_embed = self.pos_embed.to(x[self.modalities[0]].device)
        for modality in self.modalities:
            if modality in ["aerial", "spot", "aerial-flair", "naip", "planet"]:
                token = getattr(self, "_".join(["projector", modality]))(x[modality])
            elif modality.split("-")[-1] == "mono":
                token = getattr(self, "_".join(["projector", modality]))(
                    x[modality].unsqueeze(1),
                    torch.zeros(x[modality].shape[0], 1).to(x[modality].device) + 120,
                )
                token = token.view(token.shape[0], token.shape[1], -1).permute(0, 2, 1)
            else:
                if "_".join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, "_".join(["projector", modality]))(
                        x[modality],
                        x["_".join([modality, "dates"])],
                        x["_".join([modality, "mask"])],
                    )
                else:
                    token = getattr(self, "_".join(["projector", modality]))(
                        x[modality], x["_".join([modality, "dates"])]
                    )
            if self.keep_subpatch and modality == self.modality_keep:
                token, subs = self.spatial_encoder(token, modality, keep_subpatch=True)
                out["_".join(["subpatches"])] = subs.view(
                    -1, self.height * self.width, subs.shape[1], subs.shape[2]
                )
            else:
                token = self.spatial_encoder(token, modality)
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, self.height * self.width, self.embed_dim)
                tokens.append(token + self.pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(
                token.shape[0], -1, -1
            )
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        if self.keep_subpatch:
            return tokens, out
        return tokens
