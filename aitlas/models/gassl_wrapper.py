import os
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
from collections import OrderedDict
from ..base.foundation import FoundationModel
from .GASSL.gassl import MoCo, gassl_moco_resnet50
from .GASSL.gassl_geo import MoCo_geo, gassl_moco_geo_resnet50

class GASSL(FoundationModel):
    """AiTLAS wrapper class for GASSL model
    
    .. note:: Based on https://github.com/sustainlab-group/geography-aware-ssl
    """

    name = "GASSL"

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the GASSL backbone model from Zenodo repository or from a local path (if available).
        """

        # Define backbone checkpoints
        backbone_checkpoints = {
            'gassl_moco_resnet50': [
                {
                    'filename': 'moco.pth.tar', 
                    'record_id': '7379715',
                    'description': 'Baseline checkpoint pre-trained on fMoW'
                },
                {
                    'filename': 'moco_tp.pth.tar', 
                    'record_id': '7379715',
                    'description': 'Checkpoint pre-trained on fMoW with temporal positive pairs'
                }
            ],
            'gassl_moco_geo_resnet50': [
                {
                    'filename': 'moco_geo.pth.tar', 
                    'record_id': '7379715',
                    'description': 'Checkpoint pre-trained on fMoW with geo-aware sampling'
                },
                {
                    'filename': 'moco_geo+tp.pth.tar', 
                    'record_id': '7379715',
                    'description': 'Full GASSL checkpoint pre-trained on fMoW with geo-aware sampling and temporal positive pairs'
                }
            ]
        }
        
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
                            raw_checkpoint = torch.load(self.config.local_model_path, map_location='cpu')
                            checkpoint = self._clean_checkpoint(raw_checkpoint)
                            backbone = globals()[self.config.backbone_name]()
                            msg = backbone.load_state_dict(checkpoint, strict=True)
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    raw_checkpoint = torch.load(self.config.local_model_path, map_location='cpu')
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

        # This method MUST return the loaded backbone object for the parent class.
        return backbone

    def forward_features(self, x, **kwargs):
            """ Forward pass through the GASSL model to get feature embeddings.
            This method handles both MoCo and MoCo_geo backbones.
            
            Args:
                x (torch.Tensor): The input tensor of shape (N, C, H, W).
                **kwargs: Can include `return_all_embeddings`.
                    return_all_embeddings=False (default): Returns 128-dim embedding from MLP head.
                    return_all_embeddings=True: Returns 2048-dim embedding from ResNet backbone.
            """
            
            # Check if backbone is loaded
            if self.backbone is None:
                raise RuntimeError(
                    "The backbone model has not been loaded. "
                    "Please call the .load_backbone() method before the forward pass."
                )

            # Check whether 128-dim or 2048-dim embeddings should be returned
            return_all_embeddings = kwargs.get('return_all_embeddings', True)

            # Get the encoder
            encoder = self.backbone.encoder_q
            
            if not return_all_embeddings: # Return 128-dim embeddings       
                # Pass the input through the backbone (query encoder). Approach is the same for MoCo and MoCo_geo.
                embedding = encoder(x)
            else: # Return 2048-dim embeddings
                # Remove the MLP head (fully-connected layer) with Identity
                encoder = self.backbone.encoder_q
                # Store the original fully-connected layer
                original_fc = encoder.fc        
                # Replace it with an identity layer
                encoder.fc = nn.Identity()           
                # Run the forward pass
                embedding = encoder(x)         
                # Restore the original fc layer
                encoder.fc = original_fc
            
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


