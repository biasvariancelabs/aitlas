"""ChangeFormerV6: A Transformer-Based Siamese Network for Change Detection"""

import math
import warnings
from functools import partial

import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn

# Assuming BaseChangeDetection is available in your environment
from ..base import BaseChangeDetection


# -----------------------------------------------------------------------------
# Utils & Helper Functions (from models/ChangeFormerBaseNetworks.py & utils)
# -----------------------------------------------------------------------------


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=1
        )

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


# -----------------------------------------------------------------------------
# Transformer Components (from models/ChangeFormer.py)
# -----------------------------------------------------------------------------


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# -----------------------------------------------------------------------------
# Encoder & Decoder Classes (from models/ChangeFormer.py)
# -----------------------------------------------------------------------------


class EncoderTransformer_v3(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=3,
        in_chans=3,
        num_classes=2,
        embed_dims=[32, 64, 128, 256],
        num_heads=[2, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 3, 6, 18],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=patch_size,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(
        self,
        input_transform="multiple_select",
        in_index=[0, 1, 2, 3],
        align_corners=True,
        in_channels=[32, 64, 128, 256],
        embedding_dim=64,
        output_nc=2,
        decoder_softmax=False,
        feature_strides=[2, 4, 8, 16],
    ):
        super(DecoderTransformer_v3, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules
        self.diff_c4 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim
        )
        self.diff_c3 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim
        )
        self.diff_c2 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim
        )
        self.diff_c1 = conv_diff(
            in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim
        )

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc
        )
        self.make_pred_c3 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc
        )
        self.make_pred_c2 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc
        )
        self.make_pred_c1 = make_prediction(
            in_channels=self.embedding_dim, out_channels=self.output_nc
        )

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embedding_dim * len(in_channels),
                out_channels=self.embedding_dim,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.embedding_dim),
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(
            self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2
        )
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(
            self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2
        )
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(
            self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1
        )

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Softmax(dim=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder."""
        if self.input_transform == "resize_concat":
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        # outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0, 2, 1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0, 2, 1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        # p_c4 = self.make_pred_c4(_c4)
        # outputs.append(p_c4)
        _c4_up = resize(_c4, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0, 2, 1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0, 2, 1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(
            _c4, scale_factor=2, mode="bilinear"
        )
        # p_c3 = self.make_pred_c3(_c3)
        # outputs.append(p_c3)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0, 2, 1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0, 2, 1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(
            _c3, scale_factor=2, mode="bilinear"
        )
        # p_c2 = self.make_pred_c2(_c2)
        # outputs.append(p_c2)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode="bilinear", align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0, 2, 1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0, 2, 1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(
            _c2, scale_factor=2, mode="bilinear"
        )
        # p_c1 = self.make_pred_c1(_c1)
        # outputs.append(p_c1)

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)
        # Residual block
        x = self.dense_2x(x)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)

        # outputs.append(cp)

        if self.output_softmax:
            cp = self.active(cp)
            # temp = outputs
            # outputs = []
            # for pred in temp:
            # outputs.append(self.active(pred))

        # -----------------------------------------------------------------------------
        # NOTE: Deep Supervision Disabled
        # -----------------------------------------------------------------------------
        # The original ChangeFormer implementation returns a list `outputs` containing
        # predictions at multiple scales (1/4, 1/8, 1/16) for Deep Supervision loss.
        #
        # We return only the final prediction `cp` to:
        # 1. Maintain interface consistency with other models in AiTLAS.
        # 2. Adhere to the standard AiTLAS `forward()` protocol (single Tensor output).
        # 3. Avoid shape mismatches during standard validation/inference steps.
        # -----------------------------------------------------------------------------
        return cp


# -----------------------------------------------------------------------------
# Main ChangeFormerV6 Class (from models/ChangeFormer.py)
# -----------------------------------------------------------------------------


class ChangeFormerV6Model(nn.Module):
    """
    Implementation of ChangeFormerV6: A Transformer-Based Siamese Network for Change Detection.
    Based on: https://github.com/wgcban/ChangeFormer
    Original paper: https://ieeexplore.ieee.org/document/9883686
    DOI: 10.1109/IGARSS46834.2022.9883686
    """

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(ChangeFormerV6Model, self).__init__()

        # Transformer Encoder Settings
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3, 3, 4, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_v3(
            img_size=256,
            patch_size=7,
            in_chans=input_nc,
            num_classes=output_nc,
            embed_dims=self.embed_dims,
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop,
            drop_path_rate=self.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=self.depths,
            sr_ratios=[8, 4, 2, 1],
        )

        # Transformer Decoder Settings
        self.TDec_x2 = DecoderTransformer_v3(
            input_transform="multiple_select",
            in_index=[0, 1, 2, 3],
            align_corners=False,
            in_channels=self.embed_dims,
            embedding_dim=self.embedding_dim,
            output_nc=output_nc,
            decoder_softmax=decoder_softmax,
            feature_strides=[2, 4, 8, 16],
        )

    def forward(self, x1, x2):
        # Extract features (Siamese)
        fx1 = self.Tenc_x2(x1)
        fx2 = self.Tenc_x2(x2)

        # Decode and Difference
        cp = self.TDec_x2(fx1, fx2)

        return cp.contiguous()


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------


class ChangeFormerV6(BaseChangeDetection):
    """
    Wrapper for ChangeFormerV6
    """

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            warnings.warn(
                "Pretrained weights are not available for this model. "
                "Model will be initialized with random weights."
            )

        self.model = ChangeFormerV6Model(
            input_nc=3,  # self.config.in_channels
            output_nc=self.config.num_classes,
            decoder_softmax=False,  # Handled by loss or downstream
            embed_dim=256,
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)
