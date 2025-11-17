import torch
from torch import Tensor
import torch.nn.functional as F
from typing import overload

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
    interpolation: str = 'bicubic',
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