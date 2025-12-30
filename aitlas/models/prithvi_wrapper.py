import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .Prithvi.prithvi_mae import PrithviMAE, PrithviViT, prithvi_eo_v1_base, prithvi_eo_v2_tiny_tl, prithvi_eo_v2_base_tl, prithvi_eo_v2_large, prithvi_eo_v2_large_tl, prithvi_eo_v2_huge, prithvi_eo_v2_huge_tl
from aitlas.models.registries import BACKBONE_REGISTRY

BACKBONE_REGISTRY.register("Prithvi")
class Prithvi(FoundationModel):
    """AiTLAS wrapper class for Prithvi model
    
    .. note:: Based on https://github.com/NASA-IMPACT/Prithvi-EO-2.0 and https://github.com/IBM/terratorch
    """

    name = "Prithvi"

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        'prithvi_eo_v1_base': [
            {
                'filename': 'Prithvi_EO_V1_100M.pt', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M',
                'description': 'Prithvi-EO-1.0 with 100M parameters'
            },
            {
                'filename': 'sen1floods11_Prithvi_100M.pth', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11',
                'description': 'Prithvi-EO-1.0 with 100M parameters fine-tuned on Sen1Floods11 dataset'
            },
            {
                'filename': 'burn_scars_Prithvi_100M.pth', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M-burn-scar',
                'description': 'Prithvi-EO-1.0 with 100M parameters fine-tuned on HLS Burn Scar Scenes dataset'
            },
            {
                'filename': 'multi_temporal_crop_classification_Prithvi_100M.pth', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification',
                'description': 'Prithvi-EO-1.0 with 100M parameters fine-tuned on Multi-Temporal Crop Classification dataset'
            }
        ],
        'prithvi_eo_v2_tiny_tl': [
            {
                'filename': 'Prithvi_EO_V2_tiny_TL.pt', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL',
                'description': 'Prithvi-EO-2.0 based on ViT-tiny with time and location encodings'
            }
        ],
        'prithvi_eo_v2_base_tl': [
            {
                'filename': 'Prithvi_EO_V2_100M_TL.pt', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL',
                'description': 'Prithvi-EO-2.0 with 100M parameters with time and location encodings'
            }
        ],
        'prithvi_eo_v2_large': [
            {
                'filename': 'Prithvi_EO_V2_300M.pt', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M',
                'description': 'Prithvi-EO-2.0 with 300M parameters without time and location encodings'
            }
        ],
        'prithvi_eo_v2_large_tl': [
            {
                'filename': 'Prithvi_EO_V2_300M_TL.pt', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL',
                'description': 'Prithvi-EO-2.0 with 300M parameters with time and location encodings'
            }
        ],
        'prithvi_eo_v2_huge': [
            {
                'filename': 'Prithvi_EO_V2_600M.pt', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-600M',
                'description': 'Prithvi-EO-2.0 with 600M parameters without time and location encodings'
            }
        ],
        'prithvi_eo_v2_huge_tl': [
            {
                'filename': 'Prithvi_EO_V2_600M_TL.pt', 
                'repo_id': 'ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL',
                'description': 'Prithvi-EO-2.0 with 600M parameters with time and location encodings'
            }
        ],
    }

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the Prithvi backbone model from Huggingface repository or from a local path (if available).
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
        x: torch.Tensor,
        temporal_coords: None | torch.Tensor = None, 
        location_coords: None | torch.Tensor = None,
        segmentation_features: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """ Forward pass through the Prithvi model to get feature embeddings.
        
        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C, H, W)
            temporal_coords (torch.Tensor, optional): year and day-of-year info with shape (B, T, 2). Defaults to None.
            location_coords(torch.Tensor, optional): lat and lon info with shape (B, 2). Defaults to None.
            segmentation_features (bool, optional): Whether to use prepare_features_for_image_model() that prepares embedings for segmentation or not. Defaults to False.
        
        Returns:
            embedding (torch.Tensor): A feature embedding tensor from the last transformer block. If segmentation_features is True, 
                                shape is (B, D', H', W'). If False, shape is (B, 1 + N, D). True is suitable forr segmentation as a downstream task, 
                                as it provides richer embedings than False
                                B - batch size (e.g., 3 for 3 batches)
                                1+N - number of patch tokens
                                D - embedding dimension (e.g., 768 for ViT-base)
                                D' = D*T, where T is the number of time points
                                H' - patch height (e.g., 14)
                                W' - patch width (e.g., 14)
        """
        
        # Check if backbone is loaded
        if self.backbone is None:
            raise RuntimeError(
                "The backbone model has not been loaded. "
                "Please call the .load_backbone() method before the forward pass."
            )
        
        # Permute the input from (B, T, C, H, W) -> (B, C, T, H, W)
        # The former is a standard input for AiTLAS, and the latter is expected input shape for Prithvi
        x = x.permute(0, 2, 1, 3, 4)

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward_features(x, temporal_coords, location_coords, **kwargs)

        # Check if the features for segmentation should be output
        if segmentation_features == True:
            embedding = self.backbone.encoder.prepare_features_for_image_model(embedding)

        # Use the last tensor from the last (12th) transformer block
        embedding = embedding[-1]

        return embedding

for variant in Prithvi.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(Prithvi)