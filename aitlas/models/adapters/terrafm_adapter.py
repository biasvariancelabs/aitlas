import torch
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("TerraFMAdapter")
class TerraFMAdapter(BaseInputAdapter):
    def forward(self, x: torch.Tensor):
        kwargs = {}
        
        if isinstance(x, torch.Tensor):
            
            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Extract the bands from the user's config
            bands_list = getattr(self.config, "bands", [])
            
            # TerraFM requires exactly all 12 Sentinel-2 bands except B10
            allowed_bands = [
                "B01", "B02", "B03", "B04", "B05", "B06", 
                "B07", "B08", "B8A", "B09", "B11", "B12"
            ]
            
            # Default to all 12 bands if not provided or empty
            if not bands_list:
                bands_list = allowed_bands.copy()
            
            # Enforce that the provided bands exactly match the required 12 bands
            if set(bands_list) != set(allowed_bands):
                raise ValueError(
                    f"TerraFM models require exactly these 12 bands in entirety: "
                    f"{allowed_bands}. You provided: {bands_list}"
                )
            
        return x, kwargs