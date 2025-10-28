import os
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
import sys
import pytorch_lightning
from collections import OrderedDict
from ..base.foundation import FoundationModel
from .SeCo.seco import MoCoV2Module, seco_resnet18, seco_resnet50


class SeCo(FoundationModel):
    """AiTLAS wrapper class for SeCo model
    
    .. note:: Based on https://github.com/ServiceNow/seasonal-contrast
    """

    name = "SeCo"

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the SeCo backbone model from Zenodo repository or from a local path (if available).
        """

        # Define backbone checkpoints
        backbone_checkpoints = {
            'seco_resnet18': [
                {
                    'filename': 'seco_resnet18_100k.ckpt', 
                    'record_id': '4728033',
                    'description': 'SeCo with a resnet18 backbone pretrained on 100k patches'
                },
                {
                    'filename': 'seco_resnet18_1m.ckpt', 
                    'record_id': '4728033',
                    'description': 'SeCo with a resnet18 backbone pretrained on 1M patches'
                }
            ],
            'seco_resnet50': [
                {
                    'filename': 'seco_resnet50_100k.ckpt', 
                    'record_id': '4728033',
                    'description': 'SeCo with a resnet50 backbone pretrained on 100k patches'
                },
                {
                    'filename': 'seco_resnet50_1m.ckpt', 
                    'record_id': '4728033',
                    'description': 'SeCo with a resnet50 backbone pretrained on 1M patches'
                }
            ]
        }

        # A fake module mapping to satisfy the unpickler expecting an old PyTorch Lightning patch
        sys.modules['pytorch_lightning.utilities.argparse_utils'] = pytorch_lightning
        # Add the missing attribute with a placeholder function
        pytorch_lightning._gpus_arg_default = lambda: None
        
        if self.config.pretrained: # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(f"Provided local model path does not exist: {self.config.local_model_path}")
                    print("Weights will be downloaded from Zenodo instead.")
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
                            record_id = temp_checkpoint_name['record_id']
                            self._download_from_zenodo(record_id=record_id, checkpoint_name=checkpoint_name, local_model_path=self.config.local_model_path)
                            raw_checkpoint = torch.load(self.config.local_model_path, map_location='cpu', weights_only=False)
                            checkpoint = self._clean_checkpoint(raw_checkpoint)
                            backbone = globals()[self.config.backbone_name]()
                            msg = backbone.load_state_dict(checkpoint, strict=True)
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    raw_checkpoint = torch.load(self.config.local_model_path, map_location='cpu', weights_only=False)
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
                    checkpoint = self._clean_checkpoint(raw_checkpoint)
                    backbone = globals()[self.backbone_name]()
                    msg = backbone.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # Clean up the patchs
        del sys.modules['pytorch_lightning.utilities.argparse_utils']
        delattr(pytorch_lightning, '_gpus_arg_default')

        # This method MUST return the loaded backbone object for the parent class.
        return backbone

    def forward_features(self, x, **kwargs):
        """ Forward pass through the SeCo model to get feature embeddings.
        
        Args:
            x (torch.Tensor): The input tensor of shape (N, C, H, W).
        """
        
        # Check if backbone is loaded
        if self.backbone is None:
            raise RuntimeError(
                "The backbone model has not been loaded. "
                "Please call the .load_backbone() method before the forward pass."
            )

        # Pass the input through the backbone (query encoder).
        embedding = self.backbone.encoder_q(x) # 512-dim embeddings (resnet18) or 2048-dim embeddings (resnet50)
        
        return embedding

    # Internal methods
    def _download_from_zenodo(self, record_id: str, checkpoint_name: str, local_model_path: str):
        """Internal method to handle downloading files from Zenodo.
        """

        zenodo_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(zenodo_url)
        record_data = response.json()
        
        file_to_download = next(f for f in record_data['files'] if f['key'] == checkpoint_name)
        download_url = file_to_download['links']['self']
        
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(local_model_path, 'wb') as f, tqdm(unit='B', unit_scale=True, desc=checkpoint_name) as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress_bar.update(len(chunk))


    def _clean_checkpoint(self, checkpoint: dict) -> OrderedDict:
        """
        Cleans a raw checkpoint dictionary by extracting the 'state_dict' and removing the 'module.' prefix from layer keys.
        """
        # Extract the actual state dictionary
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: # Fallback for other conventions
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Clean the keys
        cleaned_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # Remove 'module.' prefix from DataParallel
                name = k[7:]
                cleaned_state_dict[name] = v
            else:
                cleaned_state_dict[k] = v
                
        return cleaned_state_dict


