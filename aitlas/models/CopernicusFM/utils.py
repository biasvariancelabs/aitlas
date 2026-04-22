from typing import overload

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, vmap


@overload
def _to_tuple(value: tuple[int, int] | int) -> tuple[int, int]: ...


@overload
def _to_tuple(value: tuple[float, float] | float) -> tuple[float, float]: ...


def _to_tuple(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, int | float):
        return (value, value)
    else:
        return value


def resize_abs_pos_embed(
    pos_embed: Tensor,
    new_size: int | tuple[int, int],
    old_size: int | tuple[int, int],
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Tensor:
    """Resize absolute position embeddings to a target resolution via interpolation.

    Adapted from https://github.com/bwconrad/flexivit. Copyright (c) 2023 Ben Conrad.

    Args:
        pos_embed: Position embeddings tensor of size [b, n, d]
        new_size: Target [height, width] of embedding
        old_size: Original [height, width] of embedding
        num_prefix_tokens: Number of non-spatial prefix tokens (e.g., cls)
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing

    Returns:
        Resized pos_embed of size [b, n', d]
    """
    new_size = _to_tuple(new_size)
    old_size = _to_tuple(old_size)
    new_ntok = new_size[0] * new_size[1]

    # Return if no resize necessary
    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix, pos_embed = (
            pos_embed[:, :num_prefix_tokens],
            pos_embed[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, pos_embed = None, pos_embed

    # Interpolate position embedding
    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(
        pos_embed, size=new_size, mode=interpolation, antialias=antialias
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    # Add back extra prefix tokens
    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)

    return pos_embed


def pi_resize_patch_embed(
    patch_embed: Tensor,
    new_patch_size: tuple[int, int],
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Tensor:
    """Resample patch embeddings to a target resolution via pseudo-inverse resizing.

    Adapted from https://github.com/bwconrad/flexivit. Copyright (c) 2023 Ben Conrad.

    Args:
        patch_embed: Patch embedding parameters of size [d, c, h, w]
        new_patch_size: Target [height, width] of embedding
        interpolation: Resize interpolation type
        antialias: Whether to apply antialiasing resizing

    Returns:
        Resized pos_embed of size [d, c h', w']
    """
    assert len(patch_embed.shape) == 4, "Patch embed kernel should be a 4D tensor"
    assert len(new_patch_size) == 2, "New patch size should only be (height, width)"

    _, _, h, w = patch_embed.shape
    old_patch_size = (h, w)

    # Return original kernel if no resize is necessary
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x: Tensor, shape: tuple[int, int]) -> Tensor:
        x = F.interpolate(
            x[None, None, ...], shape, mode=interpolation, antialias=antialias
        )
        return x[0, 0, ...]

    def calculate_pinv(
        old_shape: tuple[int, int], new_shape: tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        pinv: Tensor = torch.linalg.pinv(resize_matrix)
        return pinv

    # Calculate pseudo-inverse of resize matrix
    resize_matrix_pinv = calculate_pinv(old_patch_size, new_patch_size)
    resize_matrix_pinv = resize_matrix_pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor) -> Tensor:
        h, w = new_patch_size
        resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
        return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    patch_embed = v_resample_patch_embed(patch_embed)
    return patch_embed
