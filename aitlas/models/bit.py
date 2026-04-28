"""BIT: Bitemporal Image Transformer for Change Detection"""

import torch
from einops import rearrange
from torch import nn
from torchvision import models

from ..base import BaseChangeDetection


# -----------------------------------------------------------------------------
# 1. Custom ResNet Implementation (That supports Dilation in BasicBlock)
# -----------------------------------------------------------------------------


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockDilated(nn.Module):
    """
    Standard BasicBlock but with support for Dilation > 1.
    Source: Adapted to match BIT requirements.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlockDilated, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetDilated(models.ResNet):
    """
    Subclass of torchvision ResNet that allows using our custom BasicBlockDilated.
    """

    def __init__(self, block, layers, **kwargs):
        super().__init__(block, layers, **kwargs)


def resnet18_dilated(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ResNet-18 model that supports dilation in BasicBlock.
    Loads standard ResNet-18 weights if pretrained=True.
    """
    # [2, 2, 2, 2] is the specific recipe that makes this a ResNet-18
    model = ResNetDilated(BasicBlockDilated, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # Load official weights
        weights = models.ResNet18_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=progress)

        model.load_state_dict(state_dict)

    return model


# -----------------------------------------------------------------------------
# Helper Functions / Modules (Matches models/help_funcs.py)
# -----------------------------------------------------------------------------


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim**-0.5
        self.softmax = softmax
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, m):
        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(m), self.to_v(m)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), [q, k, v])
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1) if self.softmax else dots
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual2(
                            PreNorm2(
                                dim,
                                Cross_Attention(
                                    dim,
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                    softmax=softmax,
                                ),
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x, m):
        for attn, ff in self.layers:
            x = attn(x, m)
            x = ff(x)
        return x


# -----------------------------------------------------------------------------
# Main BIT Class
# -----------------------------------------------------------------------------


class BITModel(nn.Module):
    """
    Implementation of BIT: Bitemporal Image Transformer
    Matches default 'base_transformer_pos_s4_dd8' config from original repo.
    Based on: https://github.com/justchenhao/BIT_CD
    Original paper: https://ieeexplore.ieee.org/document/9491802
    DOI: 10.1109/TGRS.2021.3095166
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        pretrained=True,
        token_len=4,
        enc_depth=1,
        dec_depth=8,
        dim_head=64,
        decoder_dim_head=64,
    ):
        super(BITModel, self).__init__()

        # --- Backbone Configuration ---
        # 1. Use replace_stride_with_dilation to enable dilated convolutions
        #    (Crucial for Receptive Field)
        #    False, True, True -> Stride 1/8, but dilated at L3, L4
        #    Resnet18 doesn't support dilation in torchvision, so we use our custom implementation that does.
        # resnet = models.resnet50(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
        resnet = resnet18_dilated(
            pretrained=pretrained, replace_stride_with_dilation=[False, True, True]
        )

        # Patch first conv layer for arbitrary input channels
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )

            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            resnet.conv1 = new_conv

        # 2. Define layers to execute.
        #    Standard BIT uses stages_num=4, so we only use up to Layer 3.
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # 3. Resolution Adjustments
        #    Original model upsamples backbone features (1/8) to (1/4) before Transformer
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        # --- Tokenizer & Projection ---
        self.token_len = token_len
        # ResNet18 Layer 3 has 256 channels
        self.conv_pred = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1, bias=False)

        # --- Transformer ---
        dim = 32
        mlp_dim = 2 * dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, dim))

        self.transformer_encoder = Transformer(
            dim=dim,
            depth=enc_depth,
            heads=8,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=0,
        )
        self.transformer_decoder = TransformerDecoder(
            dim=dim,
            depth=dec_depth,
            heads=8,
            dim_head=decoder_dim_head,
            mlp_dim=mlp_dim,
            dropout=0,
            softmax=True,
        )

        # --- Prediction Head ---
        # Original uses TwoLayerConv2d helper function
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        )

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x_flat = x.view([b, c, -1]).contiguous()
        return torch.einsum("bln,bcn->blc", spatial_attention, x_flat)

    def forward(self, x1, x2):
        # 1. Backbone (Output 1/8)
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        # 2. Upsample to 1/4 (Matches original if_upsample_2x=True)
        f1 = self.upsamplex2(f1)
        f2 = self.upsamplex2(f2)

        # 3. Projection
        f1 = self.conv_pred(f1)
        f2 = self.conv_pred(f2)

        # 4. Tokenization (on 1/4 features)
        t1 = self._forward_semantic_tokens(f1)
        t2 = self._forward_semantic_tokens(f2)

        # 5. Transformer Encoder
        tokens = torch.cat([t1, t2], dim=1) + self.pos_embedding
        tokens = self.transformer_encoder(tokens)
        t1, t2 = tokens.chunk(2, dim=1)

        # 6. Transformer Decoder
        b, c, h, w = f1.shape

        f1_flat = rearrange(f1, "b c h w -> b (h w) c")
        f1_flat = self.transformer_decoder(f1_flat, t1)
        f1 = rearrange(f1_flat, "b (h w) c -> b c h w", h=h)

        f2_flat = rearrange(f2, "b c h w -> b (h w) c")
        f2_flat = self.transformer_decoder(f2_flat, t2)
        f2 = rearrange(f2_flat, "b (h w) c -> b c h w", h=h)

        # 7. Differencing & Classification
        # Original: Abs -> Upsample x4 (to 1/1) -> Classify
        diff = torch.abs(f1 - f2)
        diff = self.upsamplex4(diff)

        output = self.classifier(diff)

        # Ensure output is contiguous to prevent issues with focal loss
        return output.contiguous()


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------


class BIT(BaseChangeDetection):
    def __init__(self, config):
        super().__init__(config)
        self.model = BITModel(
            in_channels=3,  # self.config.in_channels,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)
