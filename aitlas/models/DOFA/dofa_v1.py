# Dynamic One-For-All (DOFA) foundation model (v1)
import json
import math
import pdb
from functools import partial, reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed
from torch.nn import Conv2d, Dropout
from torch.nn.modules.utils import _pair

from ...base import FoundationModel


class DOFA_v1(FoundationModel):
    """DOFA_v1 model implementation

    .. note:: Based on https://github.com/zhu-xlab/DOFA
    """

    name = "DOFA_v1"

    def __init__(self, config):
        """config: A dictionary-like object containing model parameters.
        Expected keys: `model_name` (e.g., 'vit_base_dofa') and
        `pretrained` (boolean)."""
        super().__init__(config)

    def load_backbone(self):
        """Loads the DOFA_v1 backbone model from torch.hub repository."""
        # Load the model from torch.hub
        backbone_name = self.config.get("backbone_name", "vit_base_dofa")
        pretrained = self.config.pretrained
        backbone = torch.hub.load("zhu-xlab/DOFA", backbone_name, pretrained=pretrained)

        # Replace the head with identity if it exists
        if hasattr(backbone, "head"):
            backbone.head = nn.Identity()

        return backbone

    def forward(self, x, wave_list):
        """Forward pass through the DOFA_v1 model."""
        # Pass the input through the backbone
        embedding = self.backbone.forward(x, wave_list)

        return embedding
