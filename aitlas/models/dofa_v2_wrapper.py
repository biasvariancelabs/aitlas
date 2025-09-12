import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .DOFA.dofa_v2 import OFAViT, vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14

class DOFA_v2(FoundationModel):
    """AiTLAS wrapper class for DOFA_v2 model
    
    .. note:: Based on https://github.com/zhu-xlab/DOFA
    """

    name = "DOFA_v2"

    def __init__(self, config):    
        super().__init__(config)

    '''def load_backbone(self):
        # Do nothing for now
        pass'''

    '''def load_backbone(self):
        """ Loads the DOFA_v2 backbone model from huggingface repository
        """
        # Set the file path annd download the weights
        file_path = os.path.dirname(os.path.abspath(__file__))
        download_path = hf_hub_download(repo_id="XShadow/DOFA", filename="DOFA_ViT_base_e100.pth", local_dir=file_path)
        print(download_path)
        # Load the model
        checkpoint = torch.load(download_path)
        backbone = vit_base_patch16()
        msg = backbone.load_state_dict(checkpoint, strict=False)
        # Move the model to GPU
        backbone = backbone.to('cuda:1')
        
        return backbone'''


    def load_backbone(self):
        """ Loads the DOFA_v2 backbone model from huggingface repository
        """

        # Define checkpoints for all backbones available on Huggingface
        backbone_checkpoints = {
            'vit_small_patch16': None, 
            'vit_base_patch16': [
                'DOFA_ViT_base_e100.pth',
                'DOFA_ViT_base_e100_full_weight.pth',
                'dofav2_vit_base_e150.pth'
            ],
            'vit_large_patch16': [
                'DOFA_ViT_large_e100.pth',
                'dofav2_vit_large_e150.pth'
            ],
            'vit_huge_patch14': None
        }
        
        if self.config.pretrained: # Load pretrained weights from a local folder
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
                            self.config.local_model_path = hf_hub_download(repo_id="XShadow/DOFA", filename=checkpoint_name, local_dir=os.path.dirname(self.config.local_model_path))
                            checkpoint = torch.load(self.config.local_model_path)
                            backbone = globals()[self.config.backbone_name]()
                            msg = backbone.load_state_dict(checkpoint, strict=False)
                            print("Successfully loaded checkpoint:", checkpoint_name)

                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    checkpoint = torch.load(self.config.local_model_path)
                    checkpoint_name = os.path.basename(self.config.local_model_path)
                    # Find the backbone name corresponding to the checkpoint
                    self.backbone_name = None
                    for name, checkpoints in backbone_checkpoints.items():
                        if checkpoints and checkpoint_name in checkpoints:
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

        return backbone