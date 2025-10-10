import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from typing import Any, Dict, List
from einops import rearrange, repeat
from ..base.foundation import FoundationModel
from .schemas import PrestoSchema
from .Presto.presto import PrestoModel, presto_default
from .Presto.utils import prepare_presto_input, INPUT_PRESTO_S2_BANDS, PRESTO_S1_BANDS, ERA5_BANDS, SRTM_BANDS


class Presto(FoundationModel):
    """AiTLAS wrapper class for Presto model
    
    .. note:: Based on https://github.com/nasaharvest/presto
    """

    name = "Presto"
    schema = PrestoSchema

    input_keys = ["s1", "s2", "era5", "srtm", "dynamic_world", "latlons", "month"]

    def __init__(self, config):
        super().__init__(config)
        self.pixel_batch_size = getattr(config, 'pixel_batch_size', 64)
        self.month = getattr(config, 'month', 6)  # Default month for the whole batch

    def load_backbone(self):
        """ Loads the Presto backbone model from Huggingface repository or from a local path (if available).
        """

        # Define backbone checkpoints
        backbone_checkpoints = {
                'presto_default': [
                {
                    'filename': 'model-bfa691d3.pth',
                    'repo_id': 'torchgeo/presto',
                    'description': 'Presto default model weights'
                }
            ]
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
                    checkpoint = torch.load(self.config.local_model_path, weights_only=False)
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
    
    def forward_features(self, inputs: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass using a list of dictionaries as input. Each dictionary should contain the keys specified in `self.input_keys`.
        
        Args:
            inputs (List[Dict[str, Any]]): A list of samples. Each sample is a
                dictionary that should contain tensors for the keys specified in
                `self.input_keys` (e.g., "s1", "s2", "latlons").
                - `s1`, `s2`, `era5`, `srtm` shape: (T*C, H, W)
                - `dynamic_world` shape: (T, H, W)
                - `latlons` shape: (2, H, W)

        Returns:
            torch.Tensor: A tensor of feature embeddings with shape (B, D, H, W),
                          where B is the batch size, D is the embedding dimension,
                          and H, W are the spatial dimensions of the input.
        """

        if self.backbone is None:
            raise RuntimeError("Backbone not loaded. Call .load_backbone() first.")

        # Stack inputs from the list of dictionaries
        stacked_data, latlons, months = {}, None, None
        if not inputs: raise ValueError("Input list cannot be empty.")

        for key in self.input_keys:
            if key in inputs[0]:
                tensor_list = [sample[key] for sample in inputs]
                stacked_tensor = torch.stack(tensor_list, dim=0)
                if key == "latlons":
                    latlons = stacked_tensor
                elif key == "month":
                    months = stacked_tensor
                else:
                    stacked_data[key] = stacked_tensor

        # Call the utility function to prepare model inputs
        x, mask, dw, months = prepare_presto_input(
            **stacked_data,
            months=months,
            default_month=self.month,
            s1_bands=PRESTO_S1_BANDS,
            s2_bands=INPUT_PRESTO_S2_BANDS,
            era5_bands=ERA5_BANDS,
            srtm_bands=SRTM_BANDS
        )
        
        b, t, h, w, _ = x.shape
        num_pixels = b * h * w

        # Reshape all tensors for pixel-wise processing
        x = rearrange(x, "b t h w c -> (b h w) t c")
        mask = rearrange(mask, "b t h w c -> (b h w) t c")
        dw = rearrange(dw, "b t h w -> (b h w) t")
        months = repeat(months, "b t -> (b h w) t", h=h, w=w)
        if latlons is not None:
            latlons = rearrange(latlons, "b c h w -> (b h w) c")

        # Process in pixel batches
        output_features = torch.zeros(num_pixels, self.backbone.encoder.embedding_size, device=x.device)
        for i in range(0, num_pixels, self.pixel_batch_size):
            end = i + self.pixel_batch_size
            latlons_b = latlons[i:end] if latlons is not None else None
            
            # The encoder call now correctly handles optional latlons
            embedding_b = self.backbone.encoder(
                x=x[i:end],
                dynamic_world=dw[i:end],
                latlons=latlons_b,
                mask=mask[i:end],
                month=months[i:end],
                eval_task=True,
            )
            output_features[i:end] = embedding_b
            
        # Reshape output back to image format (B, D, H, W)
        embedding = rearrange(output_features, "(b h w) d -> b d h w", b=b, h=h, w=w)

        return embedding 