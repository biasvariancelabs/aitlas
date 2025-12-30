import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .TerraFM import TerraFMModule, terrafm_base
from aitlas.models.registries import BACKBONE_REGISTRY

BACKBONE_REGISTRY.register("TerraFM")
class TerraFM(FoundationModel):
    """AiTLAS wrapper class for TerraFM model
    
    .. note:: Based on https://github.com/mbzuai-oryx/TerraFM
    """

    name = "TerraFM"

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        'terrafm_base': [
            {
                'filename': 'TerraFM-B.pth', 
                'repo_id': 'MBZUAI/TerraFM',
                'description': 'TerraFM foundation model with a ViT-base backbone'
            }
        ]
    }

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the TerraFM backbone model from Huggingface repository or from a local path (if available).
        """
        
        if self.config.pretrained: # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(f"Provided local model path does not exist: {self.config.local_model_path}")
                    print("Weights will be downloaded from Huggingface instead.")
                    # Check if backbone is supported
                    if self.config.backbone_name not in self.BACKBONE_CHECKPOINTS:
                        raise ValueError(f"Unsupported or missing backbone: '{self.config.backbone_name}'. Supported names are: {list(self.BACKBONE_CHECKPOINTS.keys())}")
                    else:
                        # Check if backbone has weights available
                        if self.BACKBONE_CHECKPOINTS[self.config.backbone_name] is None:
                            raise ValueError(f"No pretrained weights are available for backbone '{self.config.backbone_name}'.")
                        else: # Download the weights and load the model
                            # For now, just load the first checkpoint available for the backbone
                            temp_checkpoint_name = self.BACKBONE_CHECKPOINTS[self.config.backbone_name][0]
                            checkpoint_name = temp_checkpoint_name['filename']
                            repo_id = temp_checkpoint_name['repo_id']
                            self.config.local_model_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_name, local_dir=os.path.dirname(self.config.local_model_path))                           
                            checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                            backbone = globals()[self.config.backbone_name]()
                            msg = backbone.load_state_dict(checkpoint, strict=False)
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                    checkpoint_name = os.path.basename(self.config.local_model_path)
                    # Find the backbone name corresponding to the checkpoint
                    self.backbone_name = None
                    for name, checkpoint_list in self.BACKBONE_CHECKPOINTS.items():
                        # Ensure the list of checkpoints is not None before iterating
                        if checkpoint_list:
                            # Create a list of all filenames for the current backbone
                            filenames = [ckpt['filename'] for ckpt in checkpoint_list]
                            if checkpoint_name in filenames:
                                self.backbone_name = name
                                break
                    backbone = globals()[self.backbone_name]()
                    msg = backbone.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone


    def forward_features(
        self,
        x: Tensor | None = None,
    ) -> Tensor:
        """ Forward pass through the TerraFM model to get feature embeddings.

        Args:
            x (torch.Tensor): Tensor of input images. Shape: (B, C, H, W)
          
        Returns:
            embedding (torch.Tensor): A feature embedding tensor. Shape: (B, D)
               
        """

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward_features(x=x)

        return embedding

for variant in TerraFM.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(TerraFM)