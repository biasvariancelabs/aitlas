import os

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from aitlas.models.registries import BACKBONE_REGISTRY

from ..base.foundation import FoundationModel
from .SatMAE_plusplus.models_mae import (  # noqa: F401
    MaskedAutoencoderViT,
    satmae_plusplus_vit_large,
)
from .SatMAE_plusplus.models_mae_group_channels import (  # noqa: F401
    MaskedAutoencoderGroupChannelViT,
    satmae_plusplus_vit_large_multispectral,
)


class SatMAE_plusplus(FoundationModel):
    """AiTLAS wrapper class for SatMAE++ model

    .. note:: Based on https://github.com/techmn/satmae_pp
    """

    name = "SatMAE++"

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        "satmae_plusplus_vit_large": [
            {
                "filename": "checkpoint_ViT-L_pretrain_fmow_rgb.pth",
                "repo_id": "mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb",
                "description": "Non-temporal checkpoint pre-trained on fMoW-RGB",
            },
            {
                "filename": "checkpoint_ViT-L_finetune_fmow_rgb",
                "repo_id": "mubashir04/checkpoint_ViT-L_finetune_fmow_rgb",
                "description": "Non-temporal checkpoint fine-tuned on fMoW-RGB",
            },
        ],
        "satmae_plusplus_vit_large_multispectral": [
            {
                "filename": "checkpoint_ViT-L_pretrain_fmow_sentinel.pth",
                "repo_id": "mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel",
                "description": "Multispectral checkpoint pre-trained on fMoW-Sentinel",
            },
            {
                "filename": "checkpoint_ViT-L_finetune_fmow_sentinel.pth",
                "repo_id": "mubashir04/checkpoint_ViT-L_finetune_fmow_sentinel",
                "description": "Multispectral checkpoint fine-tuned on fMoW-Sentinel",
            },
        ],
    }

    def __init__(self, config):
        super().__init__(config)

        self.out_indices = [0]

    def load_backbone(self):
        """Loads the SatMAE++ backbone model from Huggingface repository or from a local path (if available)."""

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

    def forward_features(self, x, **kwargs):
        """Forward pass through the SatMAE++ model to get feature embeddings.
        This method handles RGB and multispectral backbones.

        Args:
            x (torch.Tensor): The input tensor.
                - For RGB/multispectral models, shape should be (B, C, H, W).

        Returns:
            embedding (torch.Tensor): The output feature embeddings of shape (B, D).

        """

        # Check if backbone is loaded
        if self.backbone is None:
            raise RuntimeError(
                "The backbone model has not been loaded. "
                "Please call the .load_backbone() method before the forward pass."
            )

        # Pass the input through the backbone
        latent, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0, **kwargs)
        # Take the cls token as the final embedding
        embedding = latent[:, 0, :]

        return embedding


for variant in SatMAE_plusplus.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(SatMAE_plusplus)
