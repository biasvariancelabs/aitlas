"""
ChangeMamba: Remote Sensing Change Detection with State Space Models.
"""

import math
import warnings
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint

from ..base import BaseChangeDetection

# -----------------------------------------------------------------------------
# Selective Scan (Pure PyTorch implementation)
# -----------------------------------------------------------------------------

def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix):
    """
    Memory-efficient selective scan.
    Processes chunk sequentially to avoid massive [L,B,G,D,N] intermediates.
    """
    L_chunk = us.shape[0]
    ys = []
    h = hprefix
    # As: [G, D, N], dts: [L, B, G, D], us: [L, B, G, D]
    # Bs: [L, B, G, N], Cs: [L, B, G, N], h: [B, G, D, N]
    As_expanded = As.unsqueeze(0).unsqueeze(0)  # [1, 1, G, D, N]
    for t in range(L_chunk):
        # dA = exp(dt * A): [1,B,G,D,1] * [1,1,G,D,N] -> [1, B, G, D, N]
        dA = (dts[t:t+1].unsqueeze(-1) * As_expanded).exp()
        # dt*u*B: [1,B,G,D,1] * [1,B,G,D,1] * [1,B,G,1,N] -> [1, B, G, D, N]
        dB = dts[t:t+1].unsqueeze(-1) * us[t:t+1].unsqueeze(-1) * Bs[t:t+1].unsqueeze(3)
        h = dA * h + dB
        # y = C * h: [1,B,G,1,N] * [1,B,G,D,N] -> [1, B, G, D, N], then sum over N
        y = (Cs[t:t+1].unsqueeze(3) * h).sum(dim=-1)  # [1, B, G, D]
        ys.append(y)
    return torch.cat(ys, dim=0), h

def selective_scan_easy(us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, chunksize=64):
    """
    Memory-efficient Pure PyTorch implementation of selective scan.
    Uses small chunks and sequential processing within chunks.
    """
    inp_dtype = us.dtype
    has_D = Ds is not None

    dts = dts.float()
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).float()
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)

    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)

    B, G, N, L = Bs.shape
    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    As = As.view(G, -1, N).float()
    Bs = Bs.permute(3, 0, 1, 2).float()
    Cs = Cs.permute(3, 0, 1, 2).float()
    Ds = Ds.view(G, -1).float() if has_D else None
    D = As.shape[1]

    oys = []
    hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)

    for i in range(0, L, chunksize):
        ys, hs = selective_scan_chunk(
            us[i:i + chunksize], dts[i:i + chunksize],
            As, Bs[i:i + chunksize], Cs[i:i + chunksize], hprefix,
        )
        oys.append(ys)
        hprefix = hs[-1]

    oys = torch.cat(oys, dim=0)

    if has_D:
        oys = oys + Ds * us

    oys = oys.permute(1, 2, 3, 0).view(B, -1, L)
    return oys.to(inp_dtype)

# -----------------------------------------------------------------------------
# Utility Functions (from classification/models/vmamba.py)
# -----------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """ LayerNorm that supports NCHW data format directly. """
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if elementwise_affine else None
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.weight is not None:
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)

class Mlp(nn.Module):
    """ Simple MLP with GeLU and Dropout. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# -----------------------------------------------------------------------------
# VSSM Backbone Components (from classification/models/vmamba.py)
# -----------------------------------------------------------------------------

class SS2D(nn.Module):
    """
    State Space 2D Module.
    Adapts the 1D Mamba SSM for 2D images by scanning in 4 directions.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: projects input to 2 * d_inner (for x and z branches)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # Depthwise convolution before SSM (captures local context)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # SSM Projections for delta, B, C
        # We need 4 copies because we scan in 4 directions (Left-Right, Right-Left, Up-Down, Down-Up)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # Stack weights for efficient computation
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        # S4/Mamba Parameters A and D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt so it's in a reasonable range
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # A is typically initialized as [1, 2, ..., N]
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D is initialized to 1
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_easy

        B, C, H, W = x.shape
        L = H * W
        K = 4

        # 1. Cross Scan: Create 4 views of the image
        # - Original flattened
        # - Transposed flattened (scans vertically)
        # - Original flipped (scans backwards)
        # - Transposed flipped (scans vertically backwards)
        x_flat = x.flatten(2, 3) # (B, C, L)
        x_trans = x.transpose(2, 3).flatten(2, 3) # (B, C, L)
        x_hwwh = torch.stack([x_flat, x_trans, x_flat.flip(-1), x_trans.flip(-1)], dim=1) # (B, K=4, C, L)
        
        # 2. Project inputs to get Delta, B, C for each view
        xs = x_hwwh
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        # Reshape for scan
        xs = xs.float().view(B, -1, L) # (B, K*D, L)
        dts = dts.contiguous().float().view(B, -1, L) # (B, K*D, L)
        Bs = Bs.float().view(B, K, -1, L) 
        Cs = Cs.float().view(B, K, -1, L)
        
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # 3. Run Selective Scan
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)

        # 4. Cross Merge: Combine the 4 scanned outputs back into an image
        inv_y = out_y.transpose(2, 3).reshape(B, K, L, -1)
        # Sum of 4 directions (original, trans, flip, trans+flip)
        inv_y = (inv_y[:, 0] +
                inv_y[:, 2].flip(1) +
                inv_y[:, 1].view(B, H, W, -1).permute(0, 2, 1, 3).flatten(1, 2) +
                inv_y[:, 3].view(B, H, W, -1).permute(0, 2, 1, 3).flatten(1, 2).flip(1))
        
        y = inv_y.transpose(1, 2).view(B, C, H, W).contiguous()
        y = self.out_norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # x: data path, z: gate path

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y = self.forward_core(x)    # Mamba SSM Scan
        # Gating with z
        y = y * F.silu(z.permute(0, 3, 1, 2)).contiguous() # silu(z) as gate
        
        # Final projection
        y = y.permute(0, 2, 3, 1).contiguous()
        out = self.out_proj(y)
        return self.dropout(out)


class VSSBlock(nn.Module):
    """
    Standard block for Vision Mamba:
    Norm -> SS2D (Mamba) -> Norm -> MLP
    """
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.ln_2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, drop=0.0)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, input: torch.Tensor):
        # Mamba Branch
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        # MLP Branch
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint and input.requires_grad:
            return checkpoint(self._forward_impl, input, use_reentrant=False)
        return self._forward_impl(input)


class VSSLayer(nn.Module):
    """ A single layer consisting of multiple VSSBlocks and optional downsampling. """
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        d_state=16,
        use_checkpoint=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                use_checkpoint=use_checkpoint,
            ) for i in range(depth)
        ])
        
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x

class PatchMerging2D(nn.Module):
    """ Merges 2x2 patches into one, increasing channels by 2x. """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        
        if (H % 2 != 0) or (W % 2 != 0):
             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
             B, H, W, C = x.shape
        
        # Split into 4 chunks (top-left, top-right, bottom-left, bottom-right)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class VSSM(nn.Module):
    """
    Visual State Space Model - VMamba (Backbone).
    Hierarchical architecture with 4 stages.
    """
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dims[0]
        self.out_indices = out_indices

        # 1. Patch Embedding (Images -> Tokens)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=True),
            Permute(0, 2, 3, 1),
            (norm_layer(dims[0]) if patch_norm else nn.Identity())
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 2. Build Layers (Stages)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Output normalization for each scale feature extraction
        for i in out_indices:
            layer = norm_layer(dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        outputs = []

        for i, layer in enumerate(self.layers):
            # Pass through blocks (VSSBlock)
            for blk in layer.blocks:
                x = blk(x)

            # Extract features before downsampling
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(x)
                # Return in BCHW format
                outputs.append(out.permute(0, 3, 1, 2).contiguous())
            else:
                outputs.append(None)

            # Downsample for next stage
            if layer.downsample is not None:
                x = layer.downsample(x)

        return [out for out in outputs if out is not None]

    def forward(self, x):
        return self.forward_features(x)


# -----------------------------------------------------------------------------
# Change Decoder (from changedetection/models/ChangeDecoder.py)
# -----------------------------------------------------------------------------

class ResBlock(nn.Module):
    """ ResNet-style block for feature smoothing. """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
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

class ChangeDecoder(nn.Module):
    """
    Decoder that captures spatio-temporal relationships between pre/post images.
    
    Structure:
    - 4 Stages (scales), matching the backbone.
    - Each stage has 3 "ST Blocks" (Spatio-Temporal Blocks) that mix features differently:
      1. Sequential: Concatenate pre/post in channel dim.
      2. Cross: Interleave pre/post pixels (like a checkerboard).
      3. Parallel: Concatenate pre/post side-by-side.
    - A "Fuse Layer" combines these representations.
    - A "Smooth Layer" refines the result.
    """
    def __init__(self, in_channels, num_classes=2, embed_dim=128, d_state=16, drop_path=0.1):
        super().__init__()

        encoder_dims = in_channels  # e.g., [96, 192, 384, 768]

        # Define blocks for all 4 scales (4 down to 1)
        # We need 3 blocks per scale for the 3 interaction types.
        
        # Scale 4 (Deepest)
        self.st_block_41 = self._make_st_block(encoder_dims[-1] * 2, embed_dim, drop_path, d_state)
        self.st_block_42 = self._make_st_block(encoder_dims[-1], embed_dim, drop_path, d_state)
        self.st_block_43 = self._make_st_block(encoder_dims[-1], embed_dim, drop_path, d_state)

        # Scale 3
        self.st_block_31 = self._make_st_block(encoder_dims[-2] * 2, embed_dim, drop_path, d_state)
        self.st_block_32 = self._make_st_block(encoder_dims[-2], embed_dim, drop_path, d_state)
        self.st_block_33 = self._make_st_block(encoder_dims[-2], embed_dim, drop_path, d_state)

        # Scale 2
        self.st_block_21 = self._make_st_block(encoder_dims[-3] * 2, embed_dim, drop_path, d_state)
        self.st_block_22 = self._make_st_block(encoder_dims[-3], embed_dim, drop_path, d_state)
        self.st_block_23 = self._make_st_block(encoder_dims[-3], embed_dim, drop_path, d_state)

        # Scale 1 (Shallowest)
        self.st_block_11 = self._make_st_block(encoder_dims[-4] * 2, embed_dim, drop_path, d_state)
        self.st_block_12 = self._make_st_block(encoder_dims[-4], embed_dim, drop_path, d_state)
        self.st_block_13 = self._make_st_block(encoder_dims[-4], embed_dim, drop_path, d_state)

        # Fusion layers: Combine 5 inputs (result of ST blocks + split components)
        self.fuse_layer_4 = self._make_fuse_layer(embed_dim)
        self.fuse_layer_3 = self._make_fuse_layer(embed_dim)
        self.fuse_layer_2 = self._make_fuse_layer(embed_dim)
        self.fuse_layer_1 = self._make_fuse_layer(embed_dim)

        # Smoothing layers
        self.smooth_layer_3 = ResBlock(embed_dim, embed_dim)
        self.smooth_layer_2 = ResBlock(embed_dim, embed_dim)
        self.smooth_layer_1 = ResBlock(embed_dim, embed_dim)

        # Classifier
        self.cls_seg = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def _make_st_block(self, in_c, out_c, drop_path, d_state):
        return nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_c, out_channels=out_c),
            Permute(0, 2, 3, 1),
            VSSBlock(hidden_dim=out_c, drop_path=drop_path, norm_layer=nn.LayerNorm, d_state=d_state, use_checkpoint=True),
            Permute(0, 3, 1, 2),
        )

    def _make_fuse_layer(self, dim):
        return nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=dim * 5, out_channels=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, pre_features, post_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        # -----------------
        # Stage I (Scale 4)
        # -----------------
        # 1. Sequential: Concatenate channels
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))

        # 2. Cross: Interleave columns (Even cols=Pre, Odd cols=Post)
        B, C, H, W = pre_feat_4.size()
        p42 = self.st_block_42(torch.stack([pre_feat_4, post_feat_4], dim=-1).reshape(B, C, H, 2 * W))

        # 3. Parallel: Side-by-side concatenation (Left=Pre, Right=Post)
        p43 = self.st_block_43(torch.cat([pre_feat_4, post_feat_4], dim=-1))

        # Fuse everything
        p4 = self.fuse_layer_4(
            torch.cat(
                [
                    p41,
                    p42[:, :, :, ::2],  # Split back out the interleaved parts
                    p42[:, :, :, 1::2],
                    p43[:, :, :, 0:W],  # Split back out the parallel parts
                    p43[:, :, :, W:],
                ],
                dim=1,
            )
        )

        # -----------------
        # Stage II (Scale 3)
        # -----------------
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()

        p32 = self.st_block_32(torch.stack([pre_feat_3, post_feat_3], dim=-1).reshape(B, C, H, 2 * W))
        p33 = self.st_block_33(torch.cat([pre_feat_3, post_feat_3], dim=-1))

        p3 = self.fuse_layer_3(
            torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1)
        )

        # Upsample previous stage and add
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        # -----------------
        # Stage III (Scale 2)
        # -----------------
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()

        p22 = self.st_block_22(torch.stack([pre_feat_2, post_feat_2], dim=-1).reshape(B, C, H, 2 * W))
        p23 = self.st_block_23(torch.cat([pre_feat_2, post_feat_2], dim=-1))

        p2 = self.fuse_layer_2(
            torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1)
        )
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        # -----------------
        # Stage IV (Scale 1)
        # -----------------
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()

        p12 = self.st_block_12(torch.stack([pre_feat_1, post_feat_1], dim=-1).reshape(B, C, H, 2 * W))
        p13 = self.st_block_13(torch.cat([pre_feat_1, post_feat_1], dim=-1))

        p1 = self.fuse_layer_1(
            torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1)
        )
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        # Final prediction
        out = self.cls_seg(p1)
        # Upsample back to original image size (since scale 1 is 1/4 size)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        return out

# -----------------------------------------------------------------------------
# Main Model: ChangeMambaBCD (from changedetection/models/Mamba_backbone.py)
# -----------------------------------------------------------------------------

class ChangeMambaBCDModel(nn.Module):
    """
    ChangeMamba: Remote Sensing Change Detection with State Space Models.
    Based on: https://github.com/ChenHongruixuan/ChangeMamba
    Original paper: https://ieeexplore.ieee.org/document/10565926
    DOI: 10.1109/TGRS.2024.3417253

    Includes a pure PyTorch fallback for 'selective_scan' to ensure compatibility 
    without compiling CUDA kernels. This means it is slower than original implementation.
    """
    def __init__(self, backbone_cfg, num_classes):
        super().__init__()
        self.backbone = VSSM(**backbone_cfg)
        
        # Get output channels from backbone config
        dims = backbone_cfg.get('dims', [96, 192, 384, 768])
        
        self.decode_head = ChangeDecoder(in_channels=dims, num_classes=num_classes)

    def forward(self, x1, x2):
        # Extract features from both images (Siamese Network)
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        
        # Decode change map
        out = self.decode_head(f1, f2)
        return out

# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------

class ChangeMamba(BaseChangeDetection):
    """
    Wrapper for ChangeMamba.
    """
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            warnings.warn(
                "Pretrained weights are not available for this model. "
                "Model will be initialized with random weights."
            )
        
        model_type = 'tiny' # self.config.model_type
        in_channels = 3 #self.config.in_channels
        
        # Configurations for different model sizes
        if model_type == 'tiny':
            print("Initializing ChangeMamba with TINY configuration")
            backbone_cfg = dict(
                in_chans=in_channels,
                dims=[96, 192, 384, 768],
                depths=[2, 2, 4, 2],
                d_state=1,
                drop_path_rate=0.2,
                use_checkpoint=True
            )
        elif model_type == 'small':
            print("Initializing ChangeMamba with SMALL configuration")
            backbone_cfg = dict(
                in_chans=in_channels,
                dims=[96, 192, 384, 768],
                depths=[2, 2, 15, 2],
                d_state=1,
                drop_path_rate=0.3,
                use_checkpoint=True
            )
        elif model_type == 'base':
            print("Initializing ChangeMamba with BASE configuration")
            backbone_cfg = dict(
                in_chans=in_channels,
                dims=[128, 256, 512, 1024],
                depths=[2, 2, 15, 2],
                d_state=1,
                drop_path_rate=0.6,
                use_checkpoint=True
            )
        else:
            raise ValueError(f"Unknown ChangeMamba model_type: {model_type}")

        self.model = ChangeMambaBCDModel(
            backbone_cfg=backbone_cfg,
            num_classes=self.config.num_classes
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)