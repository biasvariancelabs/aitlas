import os
import shutil
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .AnySat.anysat import AnySatModule, anysat_tiny, anysat_small, anysat_base

class AnySat(FoundationModel):
    """AiTLAS wrapper class for AnySat model
    
    .. note:: Based on https://github.com/gastruc/AnySat
    """

    name = "AnySat"

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the AnySat (backbone) model from HuggingFace or from a local path (if available).
        """

        # Define checkpoints for all backbones available on Huggingface
        backbone_checkpoints = {
            'anysat_tiny': None,
            'anysat_small': None,
            'anysat_base': [
                'AnySat.pth',
                'AnySat_full.pth'
            ]
        }

        if hasattr(self.config, 'flash_attn'):
            flash_attn = self.config.flash_attn
            if torch.cuda.is_available():
                pass
            else:
                flash_attn = False

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
                            checkpoint_name = backbone_checkpoints[self.config.backbone_name][0]
                            downloaded_path = hf_hub_download(repo_id="g-astruc/AnySat", filename=checkpoint_name, subfolder="models", local_dir=os.path.dirname(self.config.local_model_path))
                            # Move the file from the subfolder to the desired location
                            shutil.move(downloaded_path, self.config.local_model_path)                   
                            # Clean up the empty 'models' directory
                            os.rmdir(os.path.dirname(downloaded_path))
                            # Load the checkpoint from the corrected local path
                            checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                            state_dict = checkpoint['state_dict']
                            backbone = globals()[self.config.backbone_name](flash_attn=flash_attn)
                            msg = backbone.model.load_state_dict(state_dict, strict=True)
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                    state_dict = checkpoint['state_dict']
                    checkpoint_name = os.path.basename(self.config.local_model_path)
                    # Find the backbone name corresponding to the checkpoint
                    self.backbone_name = None
                    for name, checkpoints in backbone_checkpoints.items():
                        if checkpoints and checkpoint_name in checkpoints:
                            self.backbone_name = name
                            break
                    backbone = globals()[self.backbone_name](flash_attn=flash_attn)
                    msg = backbone.model.load_state_dict(state_dict, strict=True)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone
    
    def forward_features(self, x, patch_size, output='patch', **kwargs):
        """Extract features from the model.
        """

        # Check if backbone is loaded
        if self.backbone is None:
            raise RuntimeError(
                "The backbone model has not been loaded. "
                "Please call the .load_backbone() method before the forward pass."
            )
        
        # Pass the input through the backbone
        embedding = self.backbone(x, patch_size=patch_size, output=output, **kwargs)

        return embedding