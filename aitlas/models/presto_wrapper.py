import os
from typing import Any, Dict

import torch
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from torch import nn

from aitlas.models.registries import BACKBONE_REGISTRY

from ..base.foundation import FoundationModel
from .Presto.presto import PrestoModel, presto_default  # noqa: F401
from .Presto.utils import (
    prepare_presto_input,
)

# noqa: F401
from .schemas import PrestoSchema


class Presto(FoundationModel):
    """AiTLAS wrapper class for Presto model

    .. note:: Based on https://github.com/nasaharvest/presto
    """

    name = "Presto"
    schema = PrestoSchema

    input_keys = ["s1", "s2", "era5", "srtm", "dynamic_world", "latlons", "month"]

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        "presto_default": [
            {
                "filename": "model-f317d103.pth",
                "repo_id": "torchgeo/presto",
                "description": "Presto default model weights",
            }
        ]
    }

    def __init__(self, config):
        super().__init__(config)

        self.month = self.config.month
        self.out_indices = [0]

    def load_backbone(self):
        """Loads the Presto backbone model from Huggingface repository or from a local path (if available)."""

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

    def forward_features(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Processes a batch of B image time-series to extract feature embeddings.

        Args:
            inputs (Dict[str, Any]): A dictionary where keys are data types
                (e.g., "s1", "latlons") and values are the corresponding
                batched tensors. Data tensors should have the shape
                (B, T, C, H, W).

        Returns:
            torch.Tensor: A tensor of feature embeddings with the shape
                (B, D, H, W), where B is the number of images in the input batch.
        """

        if self.backbone is None:
            raise RuntimeError("Backbone not loaded.")
        if not inputs:
            raise ValueError("Input dictionary cannot be empty.")

        # Call the utility function with the provided dictionary
        x, dw, latlons, months = prepare_presto_input(**inputs, default_month=self.month)

        b, t, _, h, w = x.shape

        # Reshape all prepared tensors into pixel-series batches
        pixel_x = rearrange(x, "b t c h w -> (b h w) t c")
        pixel_dw = rearrange(dw, "b t h w -> (b h w) t")
        pixel_months = repeat(months, "b t -> (b h w) t", h=h, w=w)
        pixel_latlons = rearrange(latlons, "b c h w -> (b h w) c")

        # Call the encoder once with the fully prepared pixel batches
        embeddings = self.backbone.encoder(
            x=pixel_x,
            dynamic_world=pixel_dw,
            latlons=pixel_latlons,
            month=pixel_months,
            eval_task=True,
        )

        # Reshape the output back to a spatial format
        output_features = rearrange(embeddings, "(b h w) d -> b d h w", b=b, h=h, w=w)

        return output_features


for variant in Presto.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(Presto)
