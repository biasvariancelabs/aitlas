from .irpe import build_rpe, get_rpe_config
from .pos_embed import get_2d_sincos_pos_embed_with_scale

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from .utils import Mlp, DropPath

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            use_flash_attn=True
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        if use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                raise ImportError("flash-attn is not installed. Please install it with `pip install flash-attn`")
        else:
            self.flash_attn_func = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.flash_attn_func is not None:
            x = self.flash_attn_func(q, k, v, causal=False)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            flash_attn=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_flash_attn=flash_attn,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class RPEAttention(nn.Module):
    '''
    Attention with image relative position encoding
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., n_modalities=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.n_modalities = n_modalities

        # image relative position encoding
        rpe_config = get_rpe_config(
                ratio=1.9,
                method="euc",
                mode='ctx',
                shared_head=True,
                skip=1,
                rpe_on='k',
            )
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads,
                      n_modalities=n_modalities)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        if mask is None:
            height = int((N // self.n_modalities) ** .5)
        else:
            height = mask.shape[-1]

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q *= self.scale

        attn = (q @ k.transpose(-2, -1))
        # image relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pos=mask, height=height, width=height)

        # # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v

        # image relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockTransformer(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_modalities=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RPEAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
            proj_drop=drop, n_modalities=n_modalities)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class CrossRPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., 
                 proj_drop=0., num_patches=3, n_modalities=1, height=6, width=6, scale=1, modis=False):
        super().__init__()
        assert n_modalities > 0, "Number of modalities must be positive"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_patches = num_patches
        self.n_modalities = n_modalities - int(modis)
        self.scale = qk_scale or self.head_dim ** -0.5
        self.height = height
        self.width = width
        self.modis = int(modis)

        # Initialize layers
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Initialize position embedding
        self.pos_embed = get_2d_sincos_pos_embed_with_scale(
                dim, int(num_patches ** .5), cls_token=True, scale=scale, modis=modis)

        # RPE components
        rpe_config = get_rpe_config(
            ratio=1.9,
            method="euc",
            mode='ctx',
            shared_head=True,
            skip=1,
            rpe_on='k',
        )
        
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(
            rpe_config,
            head_dim=self.head_dim,
            num_heads=num_heads,
            n_modalities=1
        )
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _validate_input(self, x, mask=None):
        B, N, C = x.shape
        if mask is not None:
            assert mask.dim() == 2, f"Mask should be 2D but got shape {mask.shape}"
            assert mask.shape[0] == B, f"Mask batch size {mask.shape[0]} doesn't match input batch size {B}"
        return B, N, C

    def _compute_attention(self, q, k, v, mask=None):
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe_k is not None:
            rpe = self.rpe_k(q, height=self.height, width=self.width, pos=mask, modis=self.modis)
            if rpe is not None:
                attn += torch.cat([
                    rpe[:, :, :, :(1 + self.modis)], 
                    rpe[:, :, :, (1 + self.modis):].repeat(1, 1, 1, self.n_modalities)
                ], dim=-1)

        attn = self.attn_drop(attn.softmax(dim=-1))
        return attn

    def forward(self, x, mask=None):
        B, N, C = self._validate_input(x, mask)
        self.pos_embed = self.pos_embed.to(x.device)
        # Compute number of patches and prepare query
        num_patches = N // self.n_modalities + int(N % self.n_modalities > 0) + self.modis
        if mask is None:
            q_ = self.q_learned.expand(B, num_patches, -1) + self.pos_embed.expand(B, -1, -1)
        else:
            mask_pos = mask.unsqueeze(-1).repeat(1, 1, self.pos_embed.shape[-1])
            pos_embed = self.pos_embed.expand(B, -1, -1)
            masked_pos_embed = torch.gather(
                self.pos_embed.expand(B, -1, -1)[:, 1:], 
                dim=1, 
                index=mask_pos
            )
            pos_embed = torch.cat([pos_embed[:, :1], masked_pos_embed], dim=1)
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed

        # Reshape for attention
        q = q_.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Compute attention and output
        attn = self._compute_attention(q, k, v, mask)
        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class CrossRPEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=36, modalities=[], height=6, width=6, scale=1, modis=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossRPEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, n_modalities=len(modalities),
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches, height=height, width=width
                        , scale=scale, modis=modis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class CrossRPEAttentionMulti(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches={}, modalities={}, scales={}, release=False,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_patches = num_patches
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        if not release: 
            self.datasets = list(modalities.keys())
            self.modis = {dataset: int("modis" in modalities[dataset]) for dataset in self.datasets}
            self.len_modalities = {}
            self.pos_embed = {}
            for dataset in self.datasets:
                self.len_modalities[dataset] = len(modalities[dataset]) - int("modis" in modalities[dataset])
                for scale in scales[dataset]:
                    num_p = num_patches[dataset] // (scale * scale)
                    self.pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_scale(dim, int(num_p ** .5), 
                                                                                                scale, cls_token=True, modis=self.modis[dataset])

        # image relative position encoding
        rpe_config = get_rpe_config(
                ratio=1.9,
                method="euc",
                mode='ctx',
                shared_head=True,
                skip=1,
                rpe_on='k',
            )
        
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config,
                      head_dim=head_dim,
                      num_heads=num_heads,
                      n_modalities=1)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, dataset="", scale=1):
        B, N, C = x.shape
        pos_embed = self.pos_embed['_'.join([dataset, str(scale)])].to(x.device)
        modis = self.modis[dataset]
        # B1C -> B1H(C/H) -> BH1(C/H)
        if mask is None:
            num_patches = N // self.len_modalities[dataset] + int(N%self.len_modalities[dataset] > 0) + modis
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed.expand(B, -1, -1)
        else:
            num_patches = mask.shape[-1] + 1 + modis
            mask_pos = mask.unsqueeze(-1).repeat(1, 1, pos_embed.shape[-1])
            pos_embed_e = pos_embed.expand(B, -1, -1)
            masked_pos_embed = torch.gather(pos_embed.expand(B, -1, -1)[:, 1:], dim=1, index=mask_pos)
            pos_embed = torch.cat([pos_embed_e[:, :(1 + modis)], masked_pos_embed], dim = 1)
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed
        q = q_.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        # image relative position on keys
        if self.rpe_k is not None:
            height = int((self.num_patches[dataset] ** 0.5) / scale)
            rpe = self.rpe_k(q, height=height, width=height, pos=mask, modis=modis)
            attn += torch.cat([rpe[:, :, :, :(1+ modis)], rpe[:, :, :, (1+ modis):].repeat(1, 1, 1, self.len_modalities[dataset])], dim=-1)
            
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = torch.cat([x[:, :1], x[:, (1 + modis):]], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward_release(self, x, mask=None, n_modalities=1, modis=False, scale=1):
        B, N, C = x.shape
        num_patches = N // n_modalities + int((N - int(modis)) % n_modalities > 0) + int(modis)
        pos_embed = get_2d_sincos_pos_embed_with_scale(C, int(num_patches ** .5), scale, cls_token=True, modis=modis).to(x.device)
        # B1C -> B1H(C/H) -> BH1(C/H)
        if mask is None:
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed.expand(B, -1, -1)
        else:
            num_patches = mask.shape[-1] + 1 + modis
            mask_pos = mask.unsqueeze(-1).repeat(1, 1, pos_embed.shape[-1])
            pos_embed_e = pos_embed.expand(B, -1, -1)
            masked_pos_embed = torch.gather(pos_embed.expand(B, -1, -1)[:, 1:], dim=1, index=mask_pos)
            pos_embed = torch.cat([pos_embed_e[:, :(1 + modis)], masked_pos_embed], dim = 1)
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed
        q = q_.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        # image relative position on keys
        if self.rpe_k is not None:
            height = int((num_patches ** 0.5))
            rpe = self.rpe_k(q, height=height, width=height, pos=mask, modis=modis)
            attn += torch.cat([rpe[:, :, :, :(1+ modis)], rpe[:, :, :, (1+ modis):].repeat(1, 1, 1, n_modalities)], dim=-1)
            
        # image relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = torch.cat([x[:, :1], x[:, (1 + modis):]], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossBlockMulti(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., release=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches={}, modalities={}, scales={}):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossRPEAttentionMulti(dim, num_heads=num_heads, qkv_bias=qkv_bias, modalities=modalities, release=release,
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches, scales=scales)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, dataset="", scale=1):
        x = self.drop_path(self.attn(self.norm1(x), mask=mask, dataset=dataset, scale=scale))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def forward_release(self, x, n_modalities=1, modis=False, scale=1):
        x = self.drop_path(self.attn.forward_release(self.norm1(x), n_modalities=n_modalities, modis=modis, scale=scale))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=None, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=3, n_modalities=1, modis=False, 
                    scale=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_patches = num_patches
        self.n_modalities = n_modalities - int(modis)
        self.modis = int(modis)
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        #get_2d_sincos_pos_embed_with_scale(dim, int(num_patches ** .5), cls_token=True, scale=scale)#mdois
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        self.pos_embed = self.pos_embed.to(x.device)
        # B1C -> B1H(C/H) -> BH1(C/H)
        if mask is None:
            num_patches = N // self.n_modalities + int(N%self.n_modalities > 0) + self.modis
            q_ = self.q_learned.expand(B, num_patches, -1) + self.pos_embed.expand(B, -1, -1)
        else:
            num_patches = mask.shape[1] + 1
            mask_pos = mask.unsqueeze(-1).repeat(1, 1, self.pos_embed.shape[-1])
            pos_embed = self.pos_embed.expand(B, -1, -1)
            masked_pos_embed = torch.gather(self.pos_embed.expand(B, -1, -1)[:, 1:], dim=1, index=mask_pos)
            pos_embed = torch.cat([pos_embed[:, :1], masked_pos_embed], dim = 1)
            q_ = self.q_learned.expand(B, num_patches, -1) + pos_embed
        q = q_.reshape(B, num_patches, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        attn = self.attn_drop(attn.softmax(dim=-1))
        del q, k  # Free up memory early

        x = (attn @ v).transpose(1, 2).reshape(B, num_patches, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=None, qk_scale=None, drop=0., attn_drop=0., modis=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patches=36, modalities=[], scale=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, n_modalities=len(modalities),
                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_patches=num_patches, modis=modis, scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
