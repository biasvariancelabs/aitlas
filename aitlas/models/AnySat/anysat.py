from typing import Callable, Optional
from functools import partial

import torch
import torch.nn as nn
from .utils.utils import trunc_normal_, PatchDropout
import itertools

from .utils.utils_ViT import Block, CrossBlockMulti
from .utils.pos_embed import get_2d_sincos_pos_embed_with_scale

from .transformer import TransformerMulti
from .utils.ltae import PatchLTAEMulti
from .utils.patch_embeddings import PatchMLPMulti

class AnySatEncoder(nn.Module):
    """
    Initializes AnySat encoding module.
    Args:
        spatial_encoder (nn.Module): Neural network module for spatial encoding
        projectors (dict): Dict of all possible projectors
        modalities (dict): Dict of modalities to use
        num_patches (dict): Dict of number of patches by observation for each modality
        embed_dim (int): Embed dimension of transformer blocks. Default: 768
        depth (int): Depth of transformer blocks. Default: 12
        num_heads (int): Number of heads of transformer blocks. Default: 12
        mlp_ratio (float): MLP ratio of transformer blocks. Default: 4.
        qkv_bias (bool): Whether to use bias in QKV projection. Default: True
        qk_scale: Scale factor for QK attention. Default: None
        class_token (bool): If True, add a class token. Default: True
        pre_norm (bool): Whether to apply normalization before transformer blocks. Default: False
        drop_rate (float): Dropout rate. Default: 0.
        patch_drop_rate (float): Patch dropout rate. Default: 0.
        drop_path_rate (float): Drop path rate for transformer blocks. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        norm_layer (Optional[Callable]): Normalization layer. Default: None
        scales (dict): Dict of scales for each dataset
        keep_subpatch (bool): Whether to keep subpatch information. Default: False
        modality_keep (str): Which modality to keep subpatches for. Default: ""
        flash_attn (bool): Whether to use flash attention. Default: True
        release (bool): Whether to initialize hte model as the feature extractor. Default: False
    """
    def __init__(self,
                 spatial_encoder: nn.Module,
                 projectors: dict = {},
                 modalities: dict = {},
                 num_patches: dict = {},
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale = None,
                 class_token: bool = True,
                 pre_norm: bool = False,
                 drop_rate: float = 0.,
                 patch_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 norm_layer: Optional[Callable] = None,
                 scales: dict = {},
                 keep_subpatch: bool = False,
                 modality_keep: str = "",
                 flash_attn: bool = True,
                 release: bool = False,
                 ):
        
        super(AnySatEncoder, self).__init__()
        self.modalities = modalities

        self.num_prefix_tokens = 1 if class_token else 0
        self.embed_dim = embed_dim
        self.keep_subpatch = keep_subpatch
        self.modality_keep = modality_keep

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        if not release:
            self.datasets = list(modalities.keys())
            self.pos_embed = {}
            for dataset in self.datasets:
                for scale in scales[dataset]:
                    num_p = num_patches[dataset] // (scale * scale)
                    self.pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_scale(
                                                                        embed_dim, 
                                                                        int(num_p ** .5), 
                                                                        scale, 
                                                                        cls_token=class_token
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

        modalities_list = sorted(list(set(list(itertools.chain.from_iterable(modalities.values())))))
        for modality in modalities_list:
            if modality.split('-')[-1] == 'mono':
                m = '-'.join(modality.split('-')[:-1])
            else:
                m = modality
            setattr(self, '_'.join(['projector', modality]), projectors[m])

        self.spatial_encoder = spatial_encoder 

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth + 1)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, 
                drop_path=dpr[i], norm_layer=norm_layer, flash_attn=flash_attn) for i in range(depth)] + [CrossBlockMulti(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, modalities=modalities,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer, num_patches=num_patches,
                scales=scales, release=release)
                ])
        trunc_normal_(self.cls_token, std=.02)

    def forward_proj(self, x):
        """
        Forward function until masking used during pretraining
        """
        tokens = []
        masks = {}
        out = {}
        pos_embed = self.pos_embed['_'.join([x['dataset'], str(x['scale'])])].to(x['label'].device)
        _, N, _ = pos_embed.shape
        for modality in self.modalities[x['dataset']]:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['scale'])
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], x['scale'], x['_'.join([modality, "mask"])])
                    if modality != "modis":
                        out['_'.join(['masks', modality])] = get_mask(x['_'.join([modality, "mask"])], modality)
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], x['scale'])
            token = self.spatial_encoder(token, modality, x['dataset'], x['scale'])
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N - 1, self.embed_dim)
                out['_'.join(['tokens', modality])] = token
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        return tokens, out
    
    def forward_transformer(self, x, mask, dataset, scale):
        """
        Forward function after masking used during pretraining
        """
        pos_embed = self.pos_embed['_'.join([dataset, str(scale)])].to(x.device)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, x), dim=1)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, mask, dataset=dataset, scale=scale)
        return tokens

    def forward(self, x):
        """
        Complete forward function during training
        """
        tokens = []
        out = {}
        pos_embed = self.pos_embed['_'.join([x['dataset'], str(x['scale'])])].to(x['label'].device)
        _, N, _ = pos_embed.shape
        for modality in self.modalities[x['dataset']]:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['scale'])
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], x['scale'], x['_'.join([modality, "mask"])])
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], x['scale'])

            if self.keep_subpatch and modality == self.modality_keep:
                token, subs = self.spatial_encoder(token, modality, x['dataset'], x['scale'], keep_subpatch=True)
                out['_'.join(['subpatches'])] = subs.view(-1, N - 1, subs.shape[1], subs.shape[2])
            else:
                token = self.spatial_encoder(token, modality, x['dataset'], x['scale'])
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N - 1, self.embed_dim)
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(token.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1](tokens, dataset=x['dataset'], scale=x['scale'])
        if self.keep_subpatch:
            return tokens, out
        return tokens
    
    def forward_release(self, x, scale, output='patch', output_modality=''):
        tokens = []
        out = {}
        keep_subpatch = (output == 'dense')
        modalities = [mod for mod in x.keys() if not (mod.endswith('_dates') or mod.endswith('_mask'))]
        if keep_subpatch and output_modality == '':
            output_modality = modalities[0]
        batch_size = x[modalities[0]].shape[0]
        device = x[modalities[0]].device
        n_modalities = len(modalities)
        modis = ('modis' in modalities)
        pos_embed = None
        for modality in modalities:
            if modality == "aerial" or modality == "spot" or modality == "aerial-flair" or modality == "naip":
                token = getattr(self, '_'.join(['projector', modality]))(x[modality], scale)
            else:
                if '_'.join([modality, "mask"]) in list(x.keys()):
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], 
                        x['_'.join([modality, "dates"])], scale, x['_'.join([modality, "mask"])])
                else:
                    token = getattr(self, '_'.join(['projector', modality]))(x[modality], x['_'.join([modality, "dates"])], scale)
            
            if pos_embed is None and modality != "modis":
                B, _, C = token.shape
                N = B // batch_size
                num_patches = int(N**(1/2))
                pos_embed = get_2d_sincos_pos_embed_with_scale(C, 
                                                       num_patches, 
                                                       scale, 
                                                       cls_token=True).to(device)
            if keep_subpatch and modality == output_modality:
                token, subs = self.spatial_encoder.forward_release(token, modality, scale, keep_subpatch=True)
                out['_'.join(['subpatches'])] = subs.view(-1, N, subs.shape[1], subs.shape[2])
            else:
                token = self.spatial_encoder.forward_release(token, modality, scale)
            if modality == "modis":
                tokens.insert(0, token.unsqueeze(1))
            else:
                token = token.view(-1, N, self.embed_dim)
                tokens.append(token + pos_embed[:, 1:, :])

        tokens = torch.cat(tokens, dim=1)
        if self.cls_token is not None:
            cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(token.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)
        for blk in self.blocks[:-1]:
            tokens = blk(tokens)
        tokens = self.blocks[-1].forward_release(tokens, n_modalities=n_modalities, modis=modis, scale=scale)
        if keep_subpatch:
            tokens = tokens[:, 1:].unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)
            dense_tokens = torch.cat([tokens, out['subpatches']], dim = 3)
            B, N, P, D = dense_tokens.shape
            patch_size = int(P**(1/2))
            size = num_patches * patch_size
            dense_tokens = dense_tokens.unsqueeze(2).permute(0, 2, 4, 1, 3)
            dense_tokens = dense_tokens.view(B, 1, D, N, patch_size, patch_size)
            dense_tokens = dense_tokens.view(B, 1, D, num_patches, num_patches, patch_size, patch_size).permute(0, 1, 2, 3, 5, 4, 6)
            dense_tokens = dense_tokens.reshape(B, 1, D, size, size).flatten(0, 1).permute(0, 2, 3, 1)
            return dense_tokens
        if output == 'tile':
            return tokens[:, 0, :]
        if output == 'patch':
            return tokens[:, 1:, :].view(batch_size, num_patches, num_patches, C)
        return tokens

def get_mask(mask, modality):
    if modality in ['alos', 'l7']:
        return torch.max(mask.flatten(1, 2), dim=1).values.flatten(1, 2)
    else:
        scale = 3
        mask = mask.flatten(1, 2).unfold(2, scale, scale).unfold(3, scale, scale)
        mask = mask.flatten(2, 3).flatten(3, 4)
        mask = mask.permute(0, 2, 1, 3).flatten(2, 3)
    return torch.max(mask, dim=2).values


def _get_default_config(model_size='base'):
    """Get default configuration based on model size"""
    dim = 768 if model_size == 'base' else (512 if model_size == 'small' else 256)
    depth = 6 if model_size == 'base' else (4 if model_size == 'small' else 2)
    heads = 12 if model_size == 'base' else (8 if model_size == 'small' else 4)
    base_config = {
        'modalities': {
            'all': ['aerial', 'aerial-flair', 'spot', 'naip', 's2', 's1-asc', 's1', 'alos', 'l7', 'l8', 'modis']
            },
        'projectors': {
            'aerial': {
                'patch_size': 10,
                'in_chans': 4,
                'embed_dim': dim,
                'bias': False,
                'mlp': [dim, dim*2, dim]
                },
            'aerial-flair': {
                'patch_size': 10,
                'in_chans': 5,
                'embed_dim': dim,
                'bias': False,
                'mlp': [dim, dim*2, dim]
                },
            'spot': {
                'patch_size': 10,
                'in_chans': 3,
                'embed_dim': dim,
                'bias': False,
                'resolution': 1.0,
                'mlp': [dim, dim*2, dim]
                }, 
            'naip': {
                'patch_size': 8,
                'in_chans': 4,
                'embed_dim': dim,
                'bias': False,
                'resolution': 1.25,
                'mlp': [dim, dim*2, dim]
                },
            's2': {
                'in_channels': 10,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.0,
                'T': 367,
                'in_norm': True,
                'return_att': False,
                'positional_encoding': True,
                },
            's1-asc': {
                'in_channels': 2,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            's1': {
                'in_channels': 3,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'alos': {
                'in_channels': 3,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'l7': {
                'in_channels': 6,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'l8': {
                'in_channels': 11,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 366,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                },
            'modis': {
                'in_channels': 7,
                'n_head': 16,
                'd_k': 8,
                'mlp': [dim],
                'mlp_in': [dim//8, dim//2, dim, dim*2, dim],
                'dropout': 0.2,
                'T': 367,
                'in_norm': False,
                'return_att': False,
                'positional_encoding': True,
                'reduce_scale': 12
                },
            },
        'spatial_encoder': {
            'embed_dim': dim,
            'depth': depth,
            'num_heads': heads,
            'mlp_ratio': 4.0,
            'attn_drop_rate': 0.0,
            'drop_path_rate': 0.0,
            'modalities': {
                'all': ['aerial', 'aerial-flair', 'spot', 'naip', 's2', 's1-asc', 's1', 'alos', 'l7', 'l8', 'modis']
                },
            'scales': {},
            'input_res': {
                'aerial': 2,
                'aerial-flair': 2,
                'spot': 10,
                'naip': 10,
                's2': 10,
                's1-asc': 10,
                's1-des': 10,
                's1': 10,
                'l8': 10,
                'l7': 30,
                'alos': 30,
                'modis': 250
                }
            },
        'num_patches': {},
        'embed_dim': dim,
        'depth': depth,
        'num_heads': heads,
        'mlp_ratio': 4.0,
        'class_token': True,
        'pre_norm': False,
        'drop_rate': 0.0,
        'patch_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'attn_drop_rate': 0.0,
        'scales': {},
        'flash_attn': True,
        'release': True,
    }

    return base_config

class AnySatModule(nn.Module):
    """
    A wrapper class for the AnySatEncoder based on a model size configuration.
    """

    def __init__(self, model_size='base', flash_attn=True, **kwargs):
        super().__init__()

        self.res = {
        'aerial': 0.2,
        'aerial-flair': 0.2,
        'spot': 1.0,
        'naip': 1.25,
        's2': 10,
        's1-asc': 10,
        's1-des': 10,
        's1': 10,
        'l8': 10,
        'l7': 30,
        'alos': 30,
        }
        
        # Get the default configuration for the specified model size
        config = _get_default_config(model_size)
        config['flash_attn'] = flash_attn

        # Build projectors for each modality based on the config
        projectors = {}
        for modality in config['modalities']['all']:
            proj_config = config['projectors'][modality]
            if 'T' in proj_config:  # Time-series projector
                projectors[modality] = PatchLTAEMulti(**proj_config)
            else:  # Single image projector
                projectors[modality] = PatchMLPMulti(**proj_config)

        # Build the Spatial Encoder
        spatial_encoder = TransformerMulti(**config['spatial_encoder'])
        
        # Remove keys that are not direct inputs to AnySatEncoder to avoid errors
        config.pop('projectors', None)
        config.pop('spatial_encoder', None)
        
        # Instantiate the core model with all its components
        self.model = AnySatEncoder(
            projectors=projectors,
            spatial_encoder=spatial_encoder,
            **config
        )

    def forward(self, x, patch_size, output='patch', **kwargs):
        """
        A forward pass that calls the model's forward_release method.
        """

        assert output in ['patch', 'tile', 'dense', 'all'], "Output must be one of 'patch', 'tile', 'dense', 'all'"
        sizes = {}
        for modality in list(x.keys()):
            if modality.endswith('_dates'):
                continue
            shape = x[modality].shape
            print(shape)
            assert shape[-2] == shape[-1], "Images must be squared"
            if modality in ['s2', 's1-asc', 's1', 'alos', 'l7', 'l8', 'modis']:
                assert len(shape) == 5, f"{modality} Images must be 5D: Batch, Time, Channels, Height, Width"
            else:
                assert len(shape) == 4, f"{modality} Images must be 4D: Batch, Channels, Height, Width"
                
            if modality != 'modis':
                sizes[modality] = shape[-1] * self.res[modality]
        
        if len(sizes) >= 2:
            size_values = list(sizes.values())
            for i in range(len(size_values) - 1):
                if abs(size_values[i] - size_values[i + 1]) > 1e-10:  # Using small epsilon for float comparison
                    mod1, mod2 = list(sizes.keys())[i], list(sizes.keys())[i + 1]
                    raise ValueError(f"Modalities {mod1} and {mod2} have incompatible sizes: {size_values[i]} vs {size_values[i + 1]}")            

        return self.model.forward_release(x, scale=patch_size // 10, output=output, **kwargs)

# Factory functions for dynamic instantiation
def anysat_tiny_model(**kwargs):
    """Tiny AnySat model."""
    return AnySatModule(model_size='tiny', **kwargs)

def anysat_small_model(**kwargs):
    """Small AnySat model."""
    return AnySatModule(model_size='small', **kwargs)

def anysat_base_model(**kwargs):
    """Base AnySat model."""
    return AnySatModule(model_size='base', **kwargs)

# Set the architectures
anysat_tiny = anysat_tiny_model
anysat_small = anysat_small_model
anysat_base = anysat_base_model