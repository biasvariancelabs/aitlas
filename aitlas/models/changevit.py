"""ChangeViT: Unleashing Plain Vision Transformers for Change Detection"""

import math
from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn.init import trunc_normal_
from torchvision import models

# Assuming BaseChangeDetection is available in your environment
from ..base import BaseChangeDetection


# -----------------------------------------------------------------------------
# Utils & Helper Functions
# -----------------------------------------------------------------------------


def weight_init(module):
    """Initialize weights for the decoder modules"""
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    assert isinstance(x, int)
    return (x, x)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# -----------------------------------------------------------------------------
# Layers (from model/layers)
# -----------------------------------------------------------------------------


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
    ):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    """2D image to patch embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten_embedding=True,
    ):
        super().__init__()
        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x


class SwiGLUFFNFused(nn.Module):
    """SwiGLU FFN (Simplified version without xformers dependency for portability)"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=None,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Ensure hidden_features is a multiple of 8
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    """Vision Transformer Block"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=Attention,
        ffn_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# -----------------------------------------------------------------------------
# DINO Vision Transformer - Encoder (from model/encoder.py)
# -----------------------------------------------------------------------------


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=ViTBlock,
        ffn_layer="mlp",
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim  # embed_dim=num_features
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer == "swiglu" or ffn_layer == "swiglufused":
            ffn_layer_cls = SwiGLUFFNFused
        else:
            ffn_layer_cls = Mlp

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer_cls,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(init_weights_vit_timm)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        patch_pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    def forward(self, x):
        x = self.prepare_tokens_with_masks(x)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return x_norm


class ChangeViTEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_type="tiny",
        pretrained=False,
        embed_dim=192,
        img_size=256,
    ):
        super().__init__()

        # Initialize the ViT backbone
        self.vit = DinoVisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            block_fn=partial(ViTBlock, attn_class=Attention),
            num_register_tokens=0,
        )

        # NOTE: Weight Source Explanation
        # The original ChangeViT paper/implementation employs a hybrid approach for pretrained weights:
        # 1. 'Tiny': Uses DeiT (Data-efficient Image Transformer) weights.
        #    These are supervised/distilled weights trained on ImageNet-1k.
        # 2. 'Small': Uses DINOv2 weights.
        #    These are self-supervised weights, often superior for dense prediction tasks like change detection.
        # We strictly adhere to these specific sources to match the official implementation's performance.
        # We selected 'Tiny' as default for consistency with other models which also have pretrained weights on ImageNet-1k.
        if pretrained:
            state_dict = None
            if model_type == "tiny":
                url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
                print(f"Loading DeiT-Tiny weights from {url}...")
                checkpoint = load_state_dict_from_url(url, map_location="cpu", check_hash=True)
                # DeiT checkpoints usually have the state dict under 'model'
                state_dict = checkpoint["model"]
            elif model_type == "small":
                url = (
                    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
                )
                print(f"Loading DINOv2-Small weights from {url}...")
                checkpoint = load_state_dict_from_url(url, map_location="cpu", check_hash=True)
                # DINOv2 checkpoints often contain the state dict directly at the root level
                state_dict = checkpoint

            if state_dict is not None:
                # Remove keys that cause shape mismatches (Pos Embed & Patch Embed)
                # The original authors remove these to handle the resolution difference
                # between pretrained weights (224x224) and model input (256x256).
                for k in [
                    "pos_embed",
                    "patch_embed.proj.weight",
                    "patch_embed.proj.bias",
                ]:
                    if k in state_dict:
                        print(f"Removing key {k} from pretrained weights to avoid shape mismatch.")
                        del state_dict[k]

                # Load the remaining weights
                msg = self.vit.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained {model_type} weights. Missing keys: {msg.missing_keys}")

        # ResNet Backbone for Detail Capture (Standard ImageNet pretrained if True)
        self.resnet = models.resnet18(pretrained=pretrained)

        # Patch first conv layer of ResNet for arbitrary input channels
        if in_channels != 3:
            old_conv = self.resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )

            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            self.resnet.conv1 = new_conv

        self.drop = nn.Dropout(p=0.01)

    def detail_capture(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x2 = self.drop(self.resnet.layer1(x))  # 1/4
        x3 = self.resnet.layer2(x2)  # 1/8
        x4 = self.resnet.layer3(x3)  # 1/16
        return [x2, x3, x4]

    def forward(self, x, y):
        # ViT Branch
        v_x = self.vit(x)
        v_y = self.vit(y)

        # Reshape ViT output (B, N, C) -> (B, C, H, W)
        # Assuming 256x256 input and patch_size 16 -> 16x16 feature map
        H, W = x.shape[2] // 16, x.shape[3] // 16
        v_x = rearrange(v_x, "b (h w) c -> b c h w", h=H, w=W)
        v_y = rearrange(v_y, "b (h w) c -> b c h w", h=H, w=W)

        # ResNet Branch
        c_x = self.detail_capture(x)
        c_y = self.detail_capture(y)

        # Return [x2, x3, x4, v_x]
        return c_x + [v_x], c_y + [v_y]


# -----------------------------------------------------------------------------
# Decoder (from model/decoder.py)
# -----------------------------------------------------------------------------


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape

        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)
        kv = (
            self.kv(y)
            .reshape(B2, N2, 2, self.num_heads, C1 // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    """Block for FeatureInjector (Cross Attention + MLP)"""

    def __init__(
        self,
        dim1,
        dim2,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.attn = CrossAttention(
            dim1,
            dim2,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim1,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y)))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class FeatureInjector(nn.Module):
    def __init__(
        self,
        dim1=384,
        dim2=[64, 128, 256],
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # dim1 is ViT dim (e.g. 384), dim2 is ResNet dims (64, 128, 256)
        self.c2_c5 = DecoderBlock(
            dim1,
            dim2[0],
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )
        self.c3_c5 = DecoderBlock(
            dim1,
            dim2[1],
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )
        self.c4_c5 = DecoderBlock(
            dim1,
            dim2[2],
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )

        self.fuse = nn.Conv2d(dim1 * 3, dim1, 1, bias=False)
        weight_init(self)

    def base_forward(self, c2, c3, c4, c5):
        H, W = c5.shape[2:]
        # Flatten spatial dims for Transformer
        c2 = rearrange(c2, "b c h w -> b (h w) c")
        c3 = rearrange(c3, "b c h w -> b (h w) c")
        c4 = rearrange(c4, "b c h w -> b (h w) c")
        c5 = rearrange(c5, "b c h w -> b (h w) c")

        # Inject c5 (High level) into low level features using CrossAttention
        # Note: In DecoderBlock(dim1, dim2), x is dim1 (c5), y is dim2 (cx).
        # We want to update c5 based on c2, c3, c4 context?
        # The original code calls: _c2 = self.c2_c5(c5, c2).
        # This effectively uses c5 as Query and c2 as Key/Value, updating c5 with details from c2.
        _c2 = self.c2_c5(c5, c2)
        _c2 = rearrange(_c2, "b (h w) c -> b c h w", h=H, w=W)

        _c3 = self.c3_c5(c5, c3)
        _c3 = rearrange(_c3, "b (h w) c -> b c h w", h=H, w=W)

        _c4 = self.c4_c5(c5, c4)
        _c4 = rearrange(_c4, "b (h w) c -> b c h w", h=H, w=W)

        # Fuse the enriched high-level features
        _c5 = self.fuse(torch.cat([_c2, _c3, _c4], dim=1))
        return _c5

    def forward(self, fx, fy):
        # fx/fy: [c2, c3, c4, c5]
        _c5x = self.base_forward(fx[0], fx[1], fx[2], fx[3])
        _c5y = self.base_forward(fy[0], fy[1], fy[2], fy[3])
        return _c5x, _c5y


class ChangeViTDecoder(nn.Module):
    def __init__(self, in_dim=[64, 128, 256, 192], decay=4, num_classes=2):
        super().__init__()
        c2_channel, c3_channel, c4_channel, c5_channel = in_dim

        self.structure_enhance = FeatureInjector(dim1=c5_channel, dim2=in_dim[:-1])

        self.up_c5 = nn.Sequential(
            nn.Conv2d(c5_channel, c4_channel, 1, bias=False),
            nn.ConvTranspose2d(c4_channel, c4_channel, kernel_size=4, stride=2, padding=1),
        )
        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, c3_channel, 1, bias=False),
            nn.ConvTranspose2d(c3_channel, c3_channel, kernel_size=4, stride=2, padding=1),
        )
        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, c2_channel, 1, bias=False),
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(c2_channel, num_classes, 3, 1, padding=1, bias=False),
        )

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim * 3, dim // decay, 1, bias=False),
                    nn.BatchNorm2d(dim // decay),
                    nn.ReLU(),
                    nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(dim // decay, dim // decay, 3, 1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(dim // decay, dim, 3, 1, padding=1, bias=False),
                )
                for dim in in_dim
            ]
        )
        weight_init(self)

    def difference_modeling(self, x, y, block):
        f = torch.cat([x, y, torch.abs(x - y)], dim=1)
        f = block(f)
        return f

    def forward(self, fx, fy):
        c2x, c3x, c4x = fx[:-1]
        c2y, c3y, c4y = fy[:-1]

        # Structure Enhance: inject details into high level features
        c5x, c5y = self.structure_enhance(fx, fy)

        # Difference Modeling at each scale
        c2 = self.difference_modeling(c2x, c2y, self.mlp[0])
        c3 = self.difference_modeling(c3x, c3y, self.mlp[1])
        c4 = self.difference_modeling(c4x, c4y, self.mlp[2])
        c5 = self.difference_modeling(c5x, c5y, self.mlp[3])

        # Progressive Upsampling and Fusion
        c4f = c4 + self.up_c5(c5)
        c3f = c3 + self.up_c4(c4f)
        c2f = c2 + self.up_c3(c3f)

        pred = self.classifier(c2f)
        return pred


# -----------------------------------------------------------------------------
# Main ChangeViT Model Class (from model/trainer.py)
# -----------------------------------------------------------------------------


class ChangeViTModel(nn.Module):
    """
    Implementation of ChangeViT: Unleashing Plain Vision Transformers for Change Detection
    Based on: https://github.com/zhuduowang/ChangeViT
    Original paper: https://arxiv.org/abs/2406.12847
    DOI: 10.48550/arXiv.2406.12847
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        pretrained=True,
        model_type="tiny",
        img_size=256,
    ):
        super().__init__()
        if model_type == "tiny":
            embed_dim = 192
        elif model_type == "small":
            embed_dim = 384
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.encoder = ChangeViTEncoder(
            in_channels,
            model_type,
            pretrained=pretrained,
            embed_dim=embed_dim,
            img_size=img_size,
        )
        self.decoder = ChangeViTDecoder(in_dim=[64, 128, 256, embed_dim], num_classes=num_classes)

    def forward(self, x1, x2):
        fx, fy = self.encoder(x1, x2)
        pred = self.decoder(fx, fy)
        return pred.contiguous()


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------


class ChangeViT(BaseChangeDetection):
    """
    Wrapper for ChangeViT
    """

    def __init__(self, config):
        super().__init__(config)

        self.model = ChangeViTModel(
            in_channels=3,  # self.config.in_channels,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            model_type="tiny",  # self.config.model_type,
            img_size=256,  # self.config.img_size
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)
