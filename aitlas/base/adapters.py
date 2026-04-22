import torch
import torch.nn as nn


class BaseInputAdapter(nn.Module):
    """
    Base class for model-specific input adapters.
    Routes pure dataloader tensors into the specific formats
    and kwargs expected by foundation models.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: The pure image tensor from the dataloader.
        Returns:
            formatted_x: The primary input (Tensor, Dict, etc.)
            kwargs: Any additional arguments required by the backbone.
        """
        # Default behavior: pass-through unchanged, no extra kwargs
        return x, {}
