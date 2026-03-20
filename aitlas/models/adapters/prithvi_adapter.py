import torch
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("PrithviAdapter")
class PrithviAdapter(BaseInputAdapter):
    def forward(self, x: torch.Tensor):
        kwargs = {}
        
        if isinstance(x, torch.Tensor):
            
            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Extract the bands and backbone name from the user's config
            bands_list = getattr(self.config, "bands", [])
            backbone_name = getattr(self.config, "backbone_name", "")
            backbone_name = str(backbone_name).lower()
            
            # Prithvi standard foundation models expect exactly 6 specific Sentinel-2 bands
            allowed_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]
            
            # Default to the 6 Prithvi bands if not provided or empty
            if not bands_list:
                bands_list = allowed_bands.copy()
            
            # Enforce that the provided bands exactly match the required 6 bands
            if set(bands_list) != set(allowed_bands):
                raise ValueError(
                    f"Prithvi models require exactly these 6 bands: "
                    f"{allowed_bands}. You provided: {bands_list}"
                )
            
            # Block backbones that strictly require multi-temporal data
            if "prithvi_eo_v1_base" in backbone_name:
                raise NotImplementedError(
                    f"The backbone '{backbone_name}' requires a temporal dimension of T=3. "
                    f"Such multi-temporal datasets are currently not supported in this pipeline."
                )
            elif "prithvi_eo_v2_tiny_tl" in backbone_name:
                raise NotImplementedError(
                    f"The backbone '{backbone_name}' requires a temporal dimension of T=4. "
                    f"Such multi-temporal datasets are currently not supported in this pipeline."
                )
            
            # Prithvi expects a 5D tensor: (B, T, C, H, W)
            # Our datasets generally are a 4D tensor: (B, C, H, W), so we need to add a temporal dimension of size 1
            if x.dim() == 4:
                x = x.unsqueeze(1)
            
        return x, kwargs