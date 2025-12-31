import os
import shutil
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Any, Dict, Tuple, Union
from einops import rearrange, repeat
from ..base.foundation import FoundationModel
from .Galileo import GalileoBase, Encoder, Decoder
from .Galileo.utils import CONFIG_FILENAME, ENCODER_FILENAME, construct_galileo_input
from aitlas.models.registries import BACKBONE_REGISTRY


class Galileo(FoundationModel):
    """AiTLAS wrapper class for Galileo model
    
    .. note:: Based on https://github.com/nasaharvest/galileo
    """

    name = "Galileo"

    input_keys = ["s1", "s2", "era5", "tc", "viirs", "srtm", "dw", "wc", "landscan", "latlon"]

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        'galileo_nano': [
            {
                'filename': ENCODER_FILENAME,
                'config_name': CONFIG_FILENAME,
                'repo_id': 'nasaharvest/galileo',
                'subfolder': 'models/nano',
                'description': 'Galileo model weights for nano model'
            }
        ],
        'galileo_tiny': [
            {
                'filename': ENCODER_FILENAME,
                'config_name': CONFIG_FILENAME,
                'repo_id': 'nasaharvest/galileo',
                'subfolder': 'models/tiny',
                'description': 'Galileo model weights for tiny model'
            }
        ],
        'galileo_base': [
            {
                'filename': ENCODER_FILENAME,
                'config_name': CONFIG_FILENAME,
                'repo_id': 'nasaharvest/galileo',
                'subfolder': 'models/base',
                'description': 'Galileo model weights for base model'
            }
        ]
    }

    def __init__(self, config):
        super().__init__(config)

    def load_backbone(self):
        """ Loads the Galileo backbone model from Huggingface repository or from a local path (if available).
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
                            config_name = temp_checkpoint_name['config_name']
                            repo_id = temp_checkpoint_name['repo_id']
                            subfolder = temp_checkpoint_name['subfolder']
                            downloaded_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_name, subfolder=subfolder, local_dir=os.path.dirname(self.config.local_model_path))
                            downloaded_cfg_path = hf_hub_download(repo_id=repo_id, filename=config_name, subfolder=subfolder, local_dir=os.path.dirname(self.config.local_model_path))
                            # Move the file from the subfolder to the desired location
                            shutil.move(downloaded_path, self.config.local_model_path)
                            shutil.move(downloaded_cfg_path, os.path.join(os.path.dirname(self.config.local_model_path), config_name))                  
                            # Clean up the empty 'models' directory
                            shutil.rmtree(Path(os.path.dirname(downloaded_path)).parent)
                            # Load the checkpoint from the corrected local path
                            backbone = Encoder.load_from_folder(folder=Path(os.path.dirname(self.config.local_model_path)), device=torch.device("cpu"))
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
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
                    backbone = Encoder.load_from_folder(folder=Path(os.path.dirname(self.config.local_model_path)), device=torch.device("cpu"))
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
        inputs: Dict[str, Any], 
        patch_size: int = 8,
        average_features: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Processes a batch of B image time-series to extract feature embeddings.

        Args:
            inputs (Dict[str, Any]): A dictionary where keys are data types:
                "s1": torch.Tensor of shape (B, T, C_s1, H, W); C_s1 = 2
                "s2": torch.Tensor of shape (B, T, C_s2, H, W); C_s2 = 10
                "era5": torch.Tensor of shape (B, T, C_era5); C_era5 = 2
                "tc": torch.Tensor of shape (B, T, C_tc); C_tc = 3
                "viirs": torch.Tensor of shape (B, T, C_viirs); C_viirs = 1
                "srtm": torch.Tensor of shape (B, C_srtm, H, W); C_srtm = 2
                "dw": torch.Tensor of shape (B, C_dw, H, W); C_dw = 9
                "wc": torch.Tensor of shape (B, C_wc, H, W); C_wc = 5
                "landscan": torch.Tensor of shape (B, C_landscan); C_landscan = 1
                "latlon": torch.Tensor of shape (B, 2)
                "months": torch.Tensor of shape (B, T)
            patch_size (int): The patch size to use for processing spatial inputs.
            average_features (bool): 
                If True, returns a single averaged tensor for the batch.
                If False, returns a tuple of all raw feature embedding tensors.
            **kwargs: Additional keyword arguments, such as:
                "normalize": bool, whether to normalize the inputs (default: False)

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]:
                - If average_features=True: 
                    A single tensor of shape (B, D) representing the 
                    averaged embeddings of all unmasked tokens.
                - If average_features=False: 
                    A tuple of the 9 raw, patched embedding tensors: 
                    (s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months_out).
                    Of these, only the the first four are actual data embeddings.
                        "s_t_x": torch.Tensor of shape (B, H', W', T, C_st_g, D)
                        "sp_x": torch.Tensor of shape (B, H', W', C_sp_g, D)
                        "t_x": torch.Tensor of shape (B, T, C_t_g, D
                        "st_x": torch.Tensor of shape (B, C_st_g, D)
        """
        # Reshape the inputs from the standard AiTLAS format to Galileo expected format, 
        # e.g., (B, T, C, H, W) to (B, H, W, T, C) for space-time image data
        inputs_reshaped = {}
        for key, value in inputs.items():
            if key in ["s1", "s2"]:
                # Reshape space-time image series data
                inputs_reshaped[key] = rearrange(value, "b t c h w -> b h w t c")
            elif key in ["srtm", "dw", "wc"]:
                # Reshape space image series data
                inputs_reshaped[key] = rearrange(value, "b c h w -> b h w c")
            elif key in ["era5", "tc", "viirs", "landscan", "latlon", "months"]:
                # Do not reshape these inputs
                pass
            else:
                raise ValueError(f"Unexpected input key: {key}")
        
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded.")
        if not inputs:
            raise ValueError("Input dictionary cannot be empty.")
        
        # Call the utility function with the provided dictionary
        inputs_galileo = construct_galileo_input(**inputs_reshaped, **kwargs)

        # Call the encoder
        embedding = self.backbone.forward(*inputs_galileo, patch_size=patch_size)

        # Average he output embeddings if required
        if average_features:
            embedding = self.backbone.average_tokens(*embedding[:-1]) # Exclude the months token when calculating the average
        
        return embedding

for variant in Galileo.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(Galileo)