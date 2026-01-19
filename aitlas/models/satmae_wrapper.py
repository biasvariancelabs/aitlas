import os
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
from ..base.foundation import FoundationModel
from .SatMAE.models_mae import MaskedAutoencoderViT, satmae_vit_large
from .SatMAE.models_mae_group_channels import MaskedAutoencoderGroupChannelViT, satmae_vit_base_multispectral, satmae_vit_large_multispectral
from .SatMAE.models_mae_temporal import MaskedAutoencoderTemporalViT, satmae_vit_large_temporal
from aitlas.models.registries import BACKBONE_REGISTRY


class SatMAE(FoundationModel):
    """AiTLAS wrapper class for SatMAE model
    
    .. note:: Based on https://github.com/sustainlab-group/SatMAE
    """

    name = "SatMAE"

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        'satmae_vit_large': [
            {
                'filename': 'fmow_pretrain.pth', 
                'record_id': '7369797',
                'description': 'Non-temporal checkpoint pre-trained on fMoW'
            },
            {
                'filename': 'fmow_finetune.pth', 
                'record_id': '7369797',
                'description': 'Non-temporal checkpoint fine-tuned on fMoW'
            }
        ],
        'satmae_vit_large_multispectral': [
            {
                'filename': 'pretrain-vit-large-e199.pth', 
                'record_id': '7338613',
                'description': 'Multispectral checkpoint pre-trained on fMoW-Sentinel'
            },
            {
                'filename': 'finetune-vit-large-e7.pth', 
                'record_id': '7338613',
                'description': 'Multispectral checkpoint fine-tuned on fMoW-Sentinel'
            }
        ],
        'satmae_vit_large_temporal': [
            {
                'filename': 'pretrain_fmow_temporal.pth', 
                'record_id': '7369797',
                'description': 'Temporal checkpoint pre-trained on fMoW'
            },
            {
                'filename': 'finetune_fmow_temporal.pth', 
                'record_id': '7369797',
                'description': 'Temporal checkpoint fine-tuned on fMoW'
            }
        ],
        'satmae_vit_base_multispectral': [
            {
                'filename': 'pretrain-vit-base-e199.pth', 
                'record_id': '7338613',
                'description': 'Multispectral (base model) checkpoint pre-trained on fMoW-Sentinel'
            },
            {
                'filename': 'finetune-vit-base-e7.pth', 
                'record_id': '7338613',
                'description': 'Multispectral (base model) checkpoint fine-tuned on fMoW-Sentinel'
            }
        ]
    }

    def __init__(self, config):    
        super().__init__(config)

        self.out_indices = [0]

    def load_backbone(self):
        """ Loads the SatMAE backbone model from Zenodo repository or from a local path (if available).
        """
        
        if self.config.pretrained: # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(f"Provided local model path does not exist: {self.config.local_model_path}")
                    print("Weights will be downloaded from Zenodo instead.")
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
                            record_id = temp_checkpoint_name['record_id']
                            self._download_from_zenodo(record_id=record_id, checkpoint_name=checkpoint_name, local_model_path=self.config.local_model_path)                            
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

    def forward_features(self, x, timestamps=None, **kwargs):
        """ Forward pass through the SatMAE model to get feature embeddings.
        This method handles RGB, multispectral, and temporal backbones.
        
        Args:
            x (torch.Tensor): The input tensor.
                - For RGB/multispectral models, shape should be (B, C, H, W).
                - For temporal models, shape should be (B, T, C, H, W).
            timestamps (torch.Tensor, optional): A tensor of timestamps required
                only for the temporal model. Shape (B, T). Defaults to None.

        Returns:
            embedding (torch.Tensor): The output feature embeddings of shape (B, D).

        """
        
        # Check if backbone is loaded
        if self.backbone is None:
            raise RuntimeError(
                "The backbone model has not been loaded. "
                "Please call the .load_backbone() method before the forward pass."
            )

        # Check if the backbone is temporal
        if isinstance(self.backbone, MaskedAutoencoderTemporalViT):
            raise NotImplementedError("Loading a temporal SatMAE model is not supported due to a bug in the source code for encoding timestamps.")        
            if timestamps is None:
                raise ValueError("The temporal model requires a 'timestamps' argument.")
            
            latent, _, _ = self.backbone.forward_encoder(x, timestamps, mask_ratio=0.0, **kwargs)
        else: # Standard (RGB) or multispectral backbone
            latent, _, _ = self.backbone.forward_encoder(x, mask_ratio=0.0, **kwargs)

        # Take the cls token as the final embedding
        embedding = latent[:, 0, :]
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

for variant in SatMAE.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(SatMAE)