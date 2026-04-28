import math

import torch
from torch import nn

from .utils.pos_embed import (
    get_2d_sincos_pos_embed_with_resolution,
    get_2d_sincos_pos_embed_with_scale,
)
from .utils.utils import trunc_normal_
from .utils.utils_ViT import Block, BlockTransformer


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        input_res={},
        modalities=[],
        scale=1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = get_2d_sincos_pos_embed_with_resolution(
            embed_dim, scale, input_res, cls_token=True, modalities=modalities
        )
        # --
        self.predictor_blocks = nn.ModuleList(
            [
                BlockTransformer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=None,
                    n_modalities=1,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(embed_dim)
        # ------

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, modality, keep_subpatch=False):
        # -- concat class token to x
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # -- add positional embedding to x tokens
        x += self.predictor_pos_embed[modality].to(x.device)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        if keep_subpatch:
            return x[:, 0], x[:, 1:]

        return x[:, 0]


class TransformerMulti(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        input_res={},
        modalities={},
        scales={},
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # --
        self.input_res = input_res
        datasets = list(scales.keys())
        self.predictor_pos_embed = {}
        for dataset in datasets:
            for scale in scales[dataset]:
                self.predictor_pos_embed["_".join([dataset, str(scale)])] = (
                    get_2d_sincos_pos_embed_with_resolution(
                        embed_dim,
                        scale,
                        input_res,
                        cls_token=True,
                        modalities=modalities[dataset],
                    )
                )
        # --
        self.predictor_blocks = nn.ModuleList(
            [
                BlockTransformer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=None,
                    n_modalities=1,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, modality, dataset, scale, keep_subpatch=False):
        # -- concat class token to x
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # -- add positional embedding to x tokens
        x += self.predictor_pos_embed["_".join([dataset, str(scale)])][modality].to(x.device)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        if keep_subpatch:
            return x[:, 0], x[:, 1:]

        return x[:, 0]

    def forward_release(self, x, modality, scale, keep_subpatch=False):
        B, N, C = x.shape
        # -- concat class token to x
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # -- add positional embedding to x tokens
        x += get_2d_sincos_pos_embed_with_resolution(
            C, scale, self.input_res, cls_token=True, modalities=[modality]
        )[modality].to(x.device)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        if keep_subpatch:
            return x[:, 0], x[:, 1:]

        return x[:, 0]


class VisionTransformerPredictor(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        scale: int = 1,
        flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = get_2d_sincos_pos_embed_with_scale(
            embed_dim,
            int(num_patches**0.5),
            scale,
            cls_token=True,
        )
        # --
        self.predictor_blocks = nn.ModuleList(
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
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (
            masks_x is not None
        ), "Cannot run predictor without mask indices"

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1).to(x.device)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1).to(x.device)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x, pred_tokens], dim=1)
        # mask_final = torch.cat([torch.cat(masks_x + [masks[i]], dim=1) for i in range (len(masks))], dim=0)
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, (N_ctxt + 1) :]
        x = self.predictor_proj(x)

        return x


class VisionTransformerPredictorMulti(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        scales={},
        flash_attn: bool = True,
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        # --
        self.datasets = list(scales.keys())
        self.predictor_pos_embed = {}
        for dataset in self.datasets:
            for scale in scales[dataset]:
                num_p = num_patches[dataset] // (scale * scale)
                self.predictor_pos_embed["_".join([dataset, str(scale)])] = (
                    get_2d_sincos_pos_embed_with_scale(
                        embed_dim, int(num_p**0.5), scale, cls_token=True
                    )
                )
        # --
        self.predictor_blocks = nn.ModuleList(
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
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks, dataset, scale):
        assert (masks is not None) and (
            masks_x is not None
        ), "Cannot run predictor without mask indices"

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        pos_embed = self.predictor_pos_embed["_".join([dataset, str(scale)])].to(x.device)
        x_pos_embed = pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x, pred_tokens], dim=1)
        mask_final = torch.cat(
            [torch.cat(masks_x + [masks[i]], dim=1) for i in range(len(masks))], dim=0
        )
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, (N_ctxt + 1) :]
        x = self.predictor_proj(x)

        return x


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat(
        [torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )
    return x


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)
