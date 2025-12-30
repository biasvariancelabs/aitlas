import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence
from huggingface_hub import hf_hub_download
from .schemas import TerraMindSchema
from ..base.foundation import FoundationModel
from .TerraMind import (
    terramind_v1_tiny, 
    terramind_v1_small, 
    terramind_v1_base, 
    terramind_v1_large,
    terramind_v1_tiny_generate, 
    terramind_v1_small_generate, 
    terramind_v1_base_generate, 
    terramind_v1_large_generate,
    terramind_v1_tiny_tim, 
    terramind_v1_small_tim, 
    terramind_v1_base_tim, 
    terramind_v1_large_tim,
    checkpoint_filter_fn,
    checkpoint_filter_fn_generate,
    checkpoint_filter_fn_tim,
    select_modality_patch_embed_weights,
    PRETRAINED_BANDS
)
from aitlas.models.registries import BACKBONE_REGISTRY

BACKBONE_REGISTRY.register("TerraMind")
class TerraMind(FoundationModel):
    """AiTLAS wrapper class for TerraMind model
    
    .. note:: Based on https://github.com/terrastackai/terratorch and https://github.com/IBM/terramind
    """

    name = "TerraMind"
    schema = TerraMindSchema

    # Define backbone checkpoints
    BACKBONE_CHECKPOINTS = {
        'terramind_v1_tiny': [
            {
                'filename': 'TerraMind_v1_tiny.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-tiny',
                'description': 'TerraMind encoder with a ViT-tiny backbone'
            }
        ],
        'terramind_v1_small': [
            {
                'filename': 'TerraMind_v1_small.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-small',
                'description': 'TerraMind encoder with a ViT-small backbone'
            }
        ],
        'terramind_v1_base': [
            {
                'filename': 'TerraMind_v1_base.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-base',
                'description': 'TerraMind encoder with a ViT-base backbone'
            }
        ],
        'terramind_v1_large': [
            {
                'filename': 'TerraMind_v1_large.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-large',
                'description': 'TerraMind encoder with a ViT-large backbone'
            }
        ],
        'terramind_v1_tiny_generate': [
            {
                'filename': 'TerraMind_v1_tiny.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-tiny',
                'description': 'TerraMind any-to-any generation model with a ViT-tiny backbone'
            }
        ],
        'terramind_v1_small_generate': [
            {
                'filename': 'TerraMind_v1_small.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-small',
                'description': 'TerraMind any-to-any generation model with a ViT-small backbone'
            }
        ],
        'terramind_v1_base_generate': [
            {
                'filename': 'TerraMind_v1_base.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-base',
                'description': 'TerraMind any-to-any generation model with a ViT-base backbone'
            }
        ],
        'terramind_v1_large_generate': [
            {
                'filename': 'TerraMind_v1_large.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-large',
                'description': 'TerraMind any-to-any generation model with a ViT-large backbone'
            }
        ],
        'terramind_v1_tiny_tim': [
            {
                'filename': 'TerraMind_v1_tiny.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-tiny',
                'description': 'TerraMind Thinking in Modalities (TiM) model with a ViT-tiny backbone'
            }
        ],
        'terramind_v1_small_tim': [
            {
                'filename': 'TerraMind_v1_small.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-small',
                'description': 'TerraMind Thinking in Modalities (TiM) model with a ViT-small backbone'
            }
        ],
        'terramind_v1_base_tim': [
            {
                'filename': 'TerraMind_v1_base.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-base',
                'description': 'TerraMind Thinking in Modalities (TiM) model with a ViT-base backbone'
            }
        ],
        'terramind_v1_large_tim': [
            {
                'filename': 'TerraMind_v1_large.pt', 
                'repo_id': 'ibm-esa-geospatial/TerraMind-1.0-large',
                'description': 'TerraMind Thinking in Modalities (TiM) model with a ViT-large backbone'
            }
        ],
    }

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the TerraMind backbone model from Huggingface repository or from a local path (if available).
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
                            # Check which TerraMind model type to use
                            if 'generate' in self.config.backbone_name:
                                backbone = globals()[self.config.backbone_name](modalities=self.config.modalities, output_modalities=self.config.output_modalities, pretrained=True)
                                checkpoint = checkpoint_filter_fn_generate(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind any-to-any generation models
                            elif 'tim' in self.config.backbone_name:
                                backbone = globals()[self.config.backbone_name](modalities=self.config.modalities, tim_modalities=self.config.tim_modalities, pretrained=True)
                                checkpoint = checkpoint_filter_fn_tim(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind Thinking in Modalities (TiM) models
                            else: 
                                backbone = globals()[self.config.backbone_name](modalities=self.config.modalities)
                                checkpoint = checkpoint_filter_fn(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind
                            msg = backbone.load_state_dict(checkpoint, strict=True)
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
                    # Check which TerraMind model type to use
                    if 'generate' in self.config.backbone_name:
                        backbone = globals()[self.config.backbone_name](modalities=self.config.modalities, output_modalities=self.config.output_modalities, pretrained=True)
                        checkpoint = checkpoint_filter_fn_generate(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind any-to-any generation models
                    elif 'tim' in self.config.backbone_name:
                        backbone = globals()[self.config.backbone_name](modalities=self.config.modalities, tim_modalities=self.config.tim_modalities, pretrained=True)
                        checkpoint = checkpoint_filter_fn_tim(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind Thinking in Modalities (TiM) models
                    else: 
                        backbone = globals()[self.config.backbone_name](modalities=self.config.modalities)
                        checkpoint = checkpoint_filter_fn(checkpoint, backbone) # Additional checkpoint filtering function for TerraMind
                    msg = backbone.load_state_dict(checkpoint, strict=True)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone

    def select_input_bands(self, 
        bands: dict[str, Sequence[str]],
        pretrained_bands: dict[str, Sequence[str]] = PRETRAINED_BANDS
    ) -> None:
        """ Select input bands for each modality and update the patch embedding weights accordingly.

        Args:
            bands (dict[str, Sequence[str]]): Bands with format {<modality>: [<band names>]}

        """

        # If the backbone name contains "generate" or "tim", raise TypeError
        if "generate" in self.config.backbone_name:
            raise TypeError(
                f"The current backbone does not support selecting input bands. "
                "Please use a backbone without '_generate' in the backbone name (e.g., 'terramind_v1_tiny')."
            )
        elif "tim" in self.config.backbone_name:
            raise TypeError(
                f"The current backbone does not support selecting input bands. "
                "Please use a backbone without '_tim' in the backbone name (e.g., 'terramind_v1_tiny')."
            )

        # Update patch embeddings weights for each provided modality by selecting the pretrained weights for each band
        self.backbone = select_modality_patch_embed_weights(
            model=self.backbone,
            bands=bands,
            pretrained_bands=pretrained_bands
        )

    def forward_features(self, 
        x: dict[str, torch.Tensor] | torch.Tensor | None = None, 
        **kwargs
    ) -> list[torch.Tensor]:
        """
        Forward pass through the TerraMind model (encoder) to get feature embeddings.

        Args:
            x (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            list[torch.Tensor]: List of transformer layer outputs. Shape (B, L, D).

        """

        # If the backbone name contains "generate" or "tim", raise TypeError
        if "generate" in self.config.backbone_name:
            raise TypeError(
                f"The current backbone does not support feature embeddings. "
                "Please use a backbone without '_generate' in the backbone name (e.g., 'terramind_v1_tiny')."
            )
        elif "tim" in self.config.backbone_name:
            raise TypeError(
                f"The current backbone does not support feature embeddings. "
                "Please use a backbone without '_tim' in the backbone name (e.g., 'terramind_v1_tiny')."
            )

        if isinstance(x, dict):
            x = x.copy()

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward(d=x, **kwargs)

        return embedding

    def generate_images(self, 
        x: dict[str, torch.Tensor] | torch.Tensor | None = None,
        standardize: bool | None = True,
        verbose:  bool | None = True,
        timesteps: int | None = 50,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the TerraMind any-to-any generation model to get generated outputs.

        Args:
            x (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W). Alternatively, keyword arguments with modality=tensor.
            standardize (bool): Whether to standardize the input images before generation. Default is True.
            verbose (bool): Whether to print verbose output during generation. Default is True.
            timesteps (int): Number of timesteps for the generation process. Default is 50.

        Retrns:
            generated_images (dict, torch.Tensor): Dict of generated output images

        """
        
        # If the backbone name does not contain "generate", raise TypeError
        if "generate" not in self.config.backbone_name:
            raise TypeError(
                f"The current backbone does not support image generation. "
                "Please use a backbone with '_generate' in the backbone name (e.g., 'terramind_v1_tiny_generate')."
            )

        # Pass the input through the any-to-any generation model
        generated_images = self.backbone.forward(d=x, standardize=standardize, verbose=verbose, timesteps=timesteps, **kwargs)

        return generated_images

    def thinking_in_modalities(self, 
        x: dict[str, torch.Tensor] | torch.Tensor | None = None, 
        **kwargs
    ) -> list[torch.Tensor]:
        """
        Forward pass through the TerraMind Thinking in Modalities model to get feature embeddings.

        Args:
            x (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            list[torch.Tensor]: List of transformer layer outputs. Shape (B, L, D).

        """

        # If the backbone name contains "generate" or "tim", raise TypeError
        if "tim" not in self.config.backbone_name:
            raise TypeError(
                f"The current backbone does not support Thinking in Modalities. "
                "Please use a backbone with '_tim' in the backbone name (e.g., 'terramind_v1_tiny_tim')."
            )

        if isinstance(x, dict):
            x = x.copy()

        # Pass the input through the backbone (encoder)
        embedding = self.backbone.forward(d=x, **kwargs)

        return embedding
    
for variant in TerraMind.BACKBONE_CHECKPOINTS.keys():
    # Any-to-any generation model cannot be used as backbone
    if "generate" not in variant:
        BACKBONE_REGISTRY.register(variant)(TerraMind)