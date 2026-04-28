import os
from typing import Dict

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from aitlas.models.registries import BACKBONE_REGISTRY

from ..base.foundation import FoundationModel


class Panopticon(FoundationModel):
    """AiTLAS wrapper class for Panopticon model

    .. note:: Based on https://github.com/Panopticon-FM/panopticon and https://github.com/torchgeo
    """

    name = "Panopticon"

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        "panopticon_base": [
            {
                "filename": "panopticon_vitb14.pth",
                "repo_id": "lewaldm/panopticon",
                "description": "Panopticon foundation model with ViT-B14 backbone",
            }
        ]
    }

    def __init__(self, config):
        super().__init__(config)

        self.out_indices = [0]

    def load_backbone(self):
        """Loads the Panopticon backbone model from Huggingface repository or from a local path (if available)."""

        if self.config.pretrained:  # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(
                        f"Provided local model path does not exist: {self.config.local_model_path}"
                    )
                    print("Weights will be downloaded from Huggingface instead.")
                    # Check if backbone is supported
                    if self.config.backbone_name not in self.BACKBONE_CHECKPOINTS:
                        raise ValueError(
                            f"Unsupported or missing backbone: '{self.config.backbone_name}'. Supported names are: {list(self.BACKBONE_CHECKPOINTS.keys())}"
                        )
                    # Check if backbone has weights available
                    elif self.BACKBONE_CHECKPOINTS[self.config.backbone_name] is None:
                        raise ValueError(
                            f"No pretrained weights are available for backbone '{self.config.backbone_name}'."
                        )
                    else:  # Download the weights and load the model
                        # For now, just load the first checkpoint available for the backbone
                        temp_checkpoint_name = self.BACKBONE_CHECKPOINTS[self.config.backbone_name][
                            0
                        ]
                        checkpoint_name = temp_checkpoint_name["filename"]
                        repo_id = temp_checkpoint_name["repo_id"]
                        self.config.local_model_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=checkpoint_name,
                            local_dir=os.path.dirname(self.config.local_model_path),
                        )
                        checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                        backbone = globals()[self.config.backbone_name]()
                        msg = backbone.load_state_dict(checkpoint, strict=False)
                        print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(
                        f"Loading weights from the provided local path: {self.config.local_model_path}"
                    )
                    checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                    checkpoint_name = os.path.basename(self.config.local_model_path)
                    # Find the backbone name corresponding to the checkpoint
                    self.backbone_name = None
                    for name, checkpoint_list in self.BACKBONE_CHECKPOINTS.items():
                        # Ensure the list of checkpoints is not None before iterating
                        if checkpoint_list:
                            # Create a list of all filenames for the current backbone
                            filenames = [ckpt["filename"] for ckpt in checkpoint_list]
                            if checkpoint_name in filenames:
                                self.backbone_name = name
                                break
                    backbone = globals()[self.backbone_name]()
                    msg = backbone.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else:  # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, "head"):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone

    def forward_features(
        self, x_dict: Dict[str, torch.Tensor], dense_features: bool = False
    ) -> torch.Tensor:
        """Forward pass through the Panopticon model to get feature embeddings.

        Args:
            x_dict (dict): Dictionary of input tensors with keys:
                imgs (torch.Tensor): Tensor of shape (B, C, H, W).
                chn_ids (torch.Tensor): Tensor of shape (B, C) encoding the spectral information
                         of each channel. For optical channels, this is the wavelength in
                         nanometers. For SAR channels, this is a negative integer as outlined
                         in https://github.com/Panopticon-FM/panopticon/blob/main/dinov2/configs/data/satellites/sentinel1.yaml
            dense_features (bool, optional): Whether to return unpooled or pooled features. Defaults to False.

        Returns:
            embedding (torch.Tensor): A feature embedding tensor
                - If dense_features=False:
                    A single tensor of shape (B, D) representing the
                    pooled embeddings of all unmasked tokens.
                - If dense_features=True:
                    A single tensor of shape (B, N + 1, D) representing the
                    unpooled embeddings of all unmasked tokens
        """

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward(x_dict, dense_features=dense_features)

        return embedding


for variant in Panopticon.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(Panopticon)
