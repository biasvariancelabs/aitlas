import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Literal, Sequence
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .CopernicusFM.copernicusfm import CopernicusFMModule, copernicusfm_base, copernicusfm_large
from aitlas.models.registries import BACKBONE_REGISTRY


class CopernicusFM(FoundationModel):
    """AiTLAS wrapper class for CopernicusFM model
    
    .. note:: Based on https://github.com/zhu-xlab/Copernicus-FM and https://github.com/torchgeo
    """

    name = "CopernicusFM"

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        'copernicusfm_base': [
            {
                'filename': 'CopernicusFM_ViT_base_varlang_e100.pth', 
                'repo_id': 'wangyi111/Copernicus-FM',
                'description': 'Copernicus-FM foundation model with a ViT-base backbone'
            }
        ],
        'copernicusfm_large': [
            {
                'filename': 'CopernicusFM_ViT_large_varlang_e100.pth', 
                'repo_id': 'wangyi111/Copernicus-FM',
                'description': 'Copernicus-FM foundation model with a ViT-large backbone'
            }
        ]
    }

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the CopernicusFM backbone model from Huggingface repository or from a local path (if available).
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
        x: Tensor,
        metadata: Tensor,
        wavelengths: Sequence[float] | None = None,
        bandwidths: Sequence[float] | None = None,
        language_embed: Tensor | None = None,
        input_mode: Literal['spectral', 'variable'] = 'spectral',
        kernel_size: int | None = None,
    ) -> Tensor:
        """ Forward pass through the CopernicusFM model to get feature embeddings.

        Args:
            x (torch.Tensor): Tensor of input images with shape of (B, C, H, W).
            metadata (torch.Tensor): Longitude (degrees), latitude (degrees), timestamp
                (days since 1970/1/1), and areas (km^2) of each patch.
                Use NaN for unknown metadata. Shape: (B, 4)
            wavelengths (list[float] or torch.Tensor): Wavelengths of each spectral band (nm). Shape: (C,).
                Only used if *input_mode=='spectral'*.
            bandwidths (list[float] or torch.Tensor): Bandwidths in nm. Shape: (C,).
                Only used if *input_mode=='spectral'*.
            language_embed (torch.Tensor): Language embedding tensor from Llama 3.2 1B (length 2048).
                Only used if *input_mode=='variable'*.
            input_mode: One of 'spectral' or 'variable'.
            kernel_size (int): If provided and differs from the initialized kernel size,
                the generated patch embed kernel weights are resized accordingly.
          
        Returns:
            embedding (torch.Tensor): A feature embedding tensor with shape of (B, D).
        """

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward_features(x=x, metadata=metadata, wavelengths=wavelengths, bandwidths=bandwidths, language_embed=language_embed, input_mode=input_mode, kernel_size=kernel_size)

        return embedding
    
for variant in CopernicusFM.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(CopernicusFM)