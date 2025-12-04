import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .TerraMind import (
    terramind_v1_tiny, 
    terramind_v1_small, 
    terramind_v1_base, 
    terramind_v1_large,
    checkpoint_filter_fn,
)

class TerraMind(FoundationModel):
    """AiTLAS wrapper class for TerraMind model
    
    .. note:: Based on https://github.com/terrastackai/terratorch and https://github.com/IBM/terramind
    """

    name = "TerraMind"

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the TerraMind backbone model from Huggingface repository or from a local path (if available).
        """

        # Define backbone checkpoints
        backbone_checkpoints = {
            'terramind_v1_tiny': [
                {
                    'filename': 'TerraMind_v1_tiny.pt', 
                    'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-tiny',
                    'description': 'TerraMind foundation model with a ViT-tiny backbone'
                }
            ],
            'terramind_v1_small': [
                {
                    'filename': 'TerraMind_v1_small.pt', 
                    'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-small',
                    'description': 'TerraMind foundation model with a ViT-small backbone'
                }
            ],
            'terramind_v1_base': [
                {
                    'filename': 'TerraMind_v1_base.pt', 
                    'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-base',
                    'description': 'TerraMind foundation model with a ViT-base backbone'
                }
            ],
            'terramind_v1_large': [
                {
                    'filename': 'TerraMind_v1_large.pt', 
                    'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-large',
                    'description': 'TerraMind foundation model with a ViT-large backbone'
                }
            ],
        }
        
        if self.config.pretrained: # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(f"Provided local model path does not exist: {self.config.local_model_path}")
                    print("Weights will be downloaded from Huggingface instead.")
                    # Check if backbone is supported
                    if self.config.backbone_name not in backbone_checkpoints:
                        raise ValueError(f"Unsupported or missing backbone: '{self.config.backbone_name}'. Supported names are: {list(backbone_checkpoints.keys())}")
                    else:
                        # Check if backbone has weights available
                        if backbone_checkpoints[self.config.backbone_name] is None:
                            raise ValueError(f"No pretrained weights are available for backbone '{self.config.backbone_name}'.")
                        else: # Download the weights and load the model
                            # For now, just load the first checkpoint available for the backbone
                            temp_checkpoint_name = backbone_checkpoints[self.config.backbone_name][0]
                            checkpoint_name = temp_checkpoint_name['filename']
                            repo_id = temp_checkpoint_name['repo_id']
                            self.config.local_model_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_name, local_dir=os.path.dirname(self.config.local_model_path))                           
                            checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                            backbone = globals()[self.config.backbone_name]()
                            checkpoint = checkpoint_filter_fn(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind
                            msg = backbone.load_state_dict(checkpoint, strict=True)
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                    checkpoint_name = os.path.basename(self.config.local_model_path)
                    # Find the backbone name corresponding to the checkpoint
                    self.backbone_name = None
                    for name, checkpoint_list in backbone_checkpoints.items():
                        # Ensure the list of checkpoints is not None before iterating
                        if checkpoint_list:
                            # Create a list of all filenames for the current backbone
                            filenames = [ckpt['filename'] for ckpt in checkpoint_list]
                            if checkpoint_name in filenames:
                                self.backbone_name = name
                                break
                    backbone = globals()[self.backbone_name]()
                    checkpoint = checkpoint_filter_fn(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind
                    msg = backbone.load_state_dict(checkpoint, strict=True)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone


    def forward_features(self, 
        x: dict[str, torch.Tensor] | torch.Tensor | None = None, 
        **kwargs
    ) -> list[torch.Tensor]:
        """
        Forward pass through the TerraMind model to get feature embeddings.

        Args:
            x (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            list[torch.Tensor]: List of transformer layer outputs. Shape (B, L, D).

        """

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward(d=x, **kwargs)

        return embedding