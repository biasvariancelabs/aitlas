import os
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
import requests
from ..base.foundation import FoundationModel
from .CACo.caco import MoCoV2CACoModule, caco_resnet18, caco_resnet50


class CACo(FoundationModel):
    """AiTLAS wrapper class for CACo model
    
    .. note:: Based on https://github.com/utkarshmall13/CACo
    """

    name = "CACo"

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the CACo backbone model from Cornell University website or from a local path (if available).
        """

        # Define backbone checkpoints
        backbone_checkpoints = {
            'caco_resnet18': [
                {
                    'filename': 'resnet18_caco_geo_100k_1000.pth', 
                    'description': 'CACo with a resnet18 backbone pretrained on 100k patches'
                },
                {
                    'filename': 'resnet18_caco_geo_1m_200.pth', 
                    'description': 'CACo with a resnet18 backbone pretrained on 1M patches'
                }
            ],
            'caco_resnet50': [
                {
                    'filename': 'resnet50_caco_geo_100k_1000.pth', 
                    'description': 'CACo with a resnet50 backbone pretrained on 100k patches'
                },
                {
                    'filename': 'resnet50_caco_geo_1m_200.pth', 
                    'description': 'CACo with a resnet50 backbone pretrained on 1M patches'
                }
            ]
        }
        
        if self.config.pretrained: # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(f"Provided local model path does not exist: {self.config.local_model_path}")
                    print("Weights will be downloaded from Cornell University website instead.")
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
                            print(checkpoint_name)
                            self._download_from_cornell(checkpoint_name=checkpoint_name, local_model_path=self.config.local_model_path)
                            checkpoint = torch.load(self.config.local_model_path, map_location='cpu', weights_only=False)
                            backbone = globals()[self.config.backbone_name]()
                            msg_q = backbone.encoder_q.load_state_dict(checkpoint, strict=True) # Load weights for the query encoder
                            msg_k = backbone.encoder_k.load_state_dict(checkpoint, strict=True) # Load weights for the key encoder
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    checkpoint = torch.load(self.config.local_model_path, map_location='cpu', weights_only=False)
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
                    msg_q = backbone.encoder_q.load_state_dict(checkpoint, strict=True) # Load weights for the query encoder
                    msg_k = backbone.encoder_k.load_state_dict(checkpoint, strict=True) # Load weights for the key encoder
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone

    def forward_features(self, x, **kwargs):
            """ Forward pass through the CACo model to get feature embeddings.
            
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
    def _download_from_cornell(self, checkpoint_name: str, local_model_path: str):
        """
        Downloads a checkpoint file from the Cornell CACO website.

        Args:
            checkpoint_name: The name of the file (e.g., "resnet50_caco_geo_1m_200.pth").
            local_model_path: The local path to save the file.
        """
        
        download_url = f"https://research.cs.cornell.edu/caco/checkpoints/{checkpoint_name}"

        try:
            # Use requests to download the weights
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()  # Check for HTTP errors

                # Get total file size from headers for the progress bar
                total_size = int(r.headers.get('content-length', 0))

                # Open the local file and write in chunks with a tqdm progress bar
                with open(local_model_path, 'wb') as f, tqdm(
                    desc=checkpoint_name,
                    total=total_size,
                    unit='B',          # Unit is Bytes
                    unit_scale=True,   # Automatically scale (KB, MB, GB)
                    unit_divisor=1024, # Use 1024 for scaling
                ) as progress_bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            print(f"\nSuccessfully downloaded {checkpoint_name} to {local_model_path}")

        except requests.exceptions.HTTPError as e:
            print(f"\nHTTP Error: {e}")
            print(f"Could not find or access file at {download_url}")
            # Clean up incomplete file on error
            if os.path.exists(local_model_path):
                os.remove(local_model_path)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            if os.path.exists(local_model_path):
                os.remove(local_model_path)