import os
from typing import Sequence

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from aitlas.models.registries import BACKBONE_REGISTRY

from ..base.foundation import FoundationModel
from .CROMA.croma import CROMAModule, croma_base, croma_large  # noqa: F401


class CROMA(FoundationModel):
    """AiTLAS wrapper class for CROMA model

    .. note:: Based on https://github.com/antofuller/CROMA and https://github.com/torchgeo
    """

    name = "CROMA"

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        "croma_base": [
            {
                "filename": "CROMA_base.pt",
                "repo_id": "antofuller/CROMA",
                "description": "CROMA foundation model with a ViT-base backbone",
            }
        ],
        "croma_large": [
            {
                "filename": "CROMA_large.pt",
                "repo_id": "antofuller/CROMA",
                "description": "CROMA foundation model with a ViT-large backbone",
            }
        ],
    }

    def __init__(self, config):
        super().__init__(config)

        self.out_indices = [0]

    def load_backbone(self):
        """Loads the CROMA backbone model from Huggingface repository or from a local path (if available)."""

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
        self,
        x_sar: Tensor | None = None,
        x_optical: Tensor | None = None,
        modalities: Sequence[str] = ["sar", "optical"],
    ) -> dict[str, Tensor]:
        """Forward pass through the CROMA model to get feature embeddings.

        Args:
            x_sar (torch.Tensor): Tensor of input Sentinel-1 SAR images with shape of (B, 2, H, W).
            x_optical (torch.Tensor): Tensor of input Sentinel-2 optical images with shape of (B, 12, H, W). B10 is omitted.
            modalities (Sequence[str], optional): List of input modalities. Options: 'sar', 'optical', or both. Deafults to ['sar', 'optical'].

        Returns:
            embedding (dict[str, Tensor]): A dictionary wit the following keys:
                - "SAR_encodings": output of the radar encoder, shape (B, N, D)
                - "SAR_GAP": output of the radar FFN (after global average pooling (GAP)), shape (B, D)
                - "optical_encodings": output of the optical encoder, shape (B, N, D)
                - "optical_GAP": output of the optical FFN (after global average pooling (GAP)), shape (B, D)
                - "joint_encodings": output of the joint radar-optical encoder, shape (B, N, D)
                - "joint_GAP": global averaging pooling the joint_encodings, shape (B, D)
        """

        # Check if user provided modalities
        if set(modalities) != set(self.backbone.modalities):
            self.backbone.modalities = modalities

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward(x_sar=x_sar, x_optical=x_optical)

        return embedding


for variant in CROMA.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(CROMA)
