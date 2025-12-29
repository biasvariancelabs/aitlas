import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .DOFA.dofa_v2 import OFAViT, dofa_v1_vit_base_patch16, dofa_v1_vit_large_patch16, dofa_v2_vit_base_patch14, dofa_v2_vit_large_patch14

class DOFA_v2(FoundationModel):
    """AiTLAS wrapper class for DOFA_v2 model
    
    .. note:: Based on https://github.com/zhu-xlab/DOFA
    """

    name = "DOFA_v2"

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the DOFA_v2 backbone model from Huggingface repository or from a local path (if available).
        """

        # Define checkpoints for all backbones available on Huggingface
        backbone_checkpoints = {
            'dofa_v1_vit_base_patch16': [
                {
                    'filename': 'DOFA_ViT_base_e100.pth', 
                    'repo_id': 'XShadow/DOFA',
                    'description': 'DOFA ViT base model trained for 100 epochs'
                },
                {
                    'filename': 'DOFA_ViT_base_e100_full_weight.pth',
                    'repo_id': 'XShadow/DOFA',
                    'description': 'DOFA ViT base model trained for 100 epochs (full weights)'
                }
            ],
            'dofa_v1_vit_large_patch16': [
                {
                    'filename': 'DOFA_ViT_large_e100.pth',
                    'repo_id': 'XShadow/DOFA',
                    'description': 'DOFA ViT large model trained for 100 epochs'
                }
            ],
            'dofa_v2_vit_base_patch14': [ # DOFA_v2
                {
                    'filename': 'dofav2_vit_base_e150.pth',
                    'repo_id': 'XShadow/DOFA',
                    'description': 'DOFA_v2 ViT base model trained for 150 epochs'
                }
            ],            
            'dofa_v2_vit_large_patch14': [ # DOFA_v2
                {
                    'filename': 'dofav2_vit_large_e150.pth',
                    'repo_id': 'XShadow/DOFA',
                    'description': 'DOFA_v2 ViT large model trained for 150 epochs'
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
                            msg = backbone.load_state_dict(checkpoint, strict=False)
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    checkpoint = torch.load(self.config.local_model_path)
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
                    msg = backbone.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone

    def forward_features(self, x, wave_list):
        """ Forward pass through the DOFA_v2 model to get the feature embeddings.
        """
        
        # Check if backbone is loaded
        if self.backbone is None:
            raise RuntimeError(
                "The backbone model has not been loaded. "
                "Please call the .load_backbone() method before the forward pass."
            )

        # Pass the input through the backbone
        embedding = self.backbone.forward_features(x, wave_list)

        return embedding