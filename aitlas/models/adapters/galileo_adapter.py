import torch
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY

@ADAPTER_REGISTRY.register("GalileoAdapter")
class GalileoAdapter(BaseInputAdapter):
    def forward(self, x):
        
        if isinstance(x, torch.Tensor):

            # Replace NaNs and Infs
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Check the config for S1 and S2 bands
            has_s1 = hasattr(self.config, "bands_s1")
            has_s2 = hasattr(self.config, "bands_s2")
            
            # Galileo expects exactly 2 S1 bands (VV, VH) and 10 S2 bands (no B1, B9, B10)
            GALILEO_S1_BANDS = ["VV", "VH"]
            GALILEO_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
            
            # Get the band list from the config 
            bands_s1 = getattr(self.config, "bands_s1", []) or GALILEO_S1_BANDS
            bands_s2 = getattr(self.config, "bands_s2", []) or GALILEO_S2_BANDS
            
            # Calculate expected slice sizes
            num_s1 = len(bands_s1) if has_s1 else 0
            num_s2 = len(bands_s2) if has_s2 else 0
            expected_channels = num_s1 + num_s2
            actual_channels = x.shape[1] if x.dim() >= 2 else 0
            
            if actual_channels < expected_channels:
                raise ValueError(
                    f"Config expects {expected_channels} channels "
                    f"but the input tensor only has {actual_channels} channels."
                )

            # Ensure S1 config matches Galileo's requirement
            if has_s1 and set(bands_s1) != set(GALILEO_S1_BANDS):
                raise ValueError(
                    f"Galileo requires exactly these 2 Sentinel-1 bands: {GALILEO_S1_BANDS}. "
                    f"Your config provided: {bands_s1}"
                )
            
            # Ensure S2 config matches Galileo's requirement
            if has_s2 and set(bands_s2) != set(GALILEO_S2_BANDS):
                raise ValueError(
                    f"Galileo requires exactly these 10 Sentinel-2 bands: {GALILEO_S2_BANDS}. "
                    f"Your config provided: {bands_s2}"
                )

            # Slice modalities
            inputs_dict = {}
            
            if has_s1:
                print(f"Config includes Sentinel-1 bands: {bands_s1}")
                # Slice S1: (B, C_s1, H, W) -> unsqueeze -> (B, 1, C_s1, H, W) where T=1
                x_s1 = x[:, :num_s1, ...]
                inputs_dict["s1"] = x_s1.unsqueeze(1)
                
            if has_s2:
                print(f"Config includes Sentinel-2 bands: {bands_s2}")
                # Slice S2: (B, C_s2, H, W) -> unsqueeze -> (B, 1, C_s2, H, W) where T=1
                x_s2 = x[:, num_s1 : num_s1 + num_s2, ...]
                inputs_dict["s2"] = x_s2.unsqueeze(1)

            # Inject the required keyword arguments for Galileo
            kwargs = {
                "inputs": inputs_dict
            }

        # Return None as the primary 'x' to force kwargs routing
        return None, kwargs