import torch
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("PrestoAdapter")
class PrestoAdapter(BaseInputAdapter):
    def forward(self, x):
        
        if isinstance(x, torch.Tensor):

            # Replace NaNs and Infs
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Get actual batch size safely
            batch_size = x.shape[0]

            # Presto expects exactly 2 S1 bands (VV, VH) and 10 S2 bands (no B1, B9 and B10)
            PRESTO_S1_BANDS = ["VV", "VH"]
            PRESTO_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

            # Check config flags
            has_s1 = hasattr(self.config, "bands_s1")
            has_s2 = hasattr(self.config, "bands_s2")
            
            # Fetch the band lists (fallback to standard defaults if empty)
            bands_s1 = getattr(self.config, "bands_s1", []) or PRESTO_S1_BANDS
            bands_s2 = getattr(self.config, "bands_s2", []) or PRESTO_S2_BANDS
            
            # Ensure config matches Presto's requirements
            if has_s1 and set(bands_s1) != set(PRESTO_S1_BANDS):
                raise ValueError(f"Presto requires exactly these S1 bands: {PRESTO_S1_BANDS}. Got: {bands_s1}")
            if has_s2 and set(bands_s2) != set(PRESTO_S2_BANDS):
                raise ValueError(f"Presto requires exactly these S2 bands: {PRESTO_S2_BANDS}. Got: {bands_s2}")

            # Calculate slice sizes
            num_s1 = len(bands_s1) if has_s1 else 0
            num_s2 = len(bands_s2) if has_s2 else 0
            
            expected_channels = num_s1 + num_s2
            actual_channels = x.shape[1]
            
            if actual_channels < expected_channels:
                raise ValueError(
                    f"Config expects {expected_channels} channels "
                    f"but the input tensor only has {actual_channels} channels."
                )

            # Slice modalities
            inputs_dict = {}
            current_idx = 0
            
            if has_s1:
                # Slice -> Unsqueeze temporal dim -> Shape: (B, 1, 2, H, W)
                inputs_dict["s1"] = x[:, current_idx : current_idx + num_s1, ...].unsqueeze(1)
                current_idx += num_s1
                
            if has_s2:
                # Slice -> Unsqueeze temporal dim -> Shape: (B, 1, 10, H, W)
                inputs_dict["s2"] = x[:, current_idx : current_idx + num_s2, ...].unsqueeze(1)
                

            # Extract spatial dimensions
            H, W = x.shape[-2], x.shape[-1]
            
            # Prepare dummy metadata
            # latlons: Shape (B, 2, H, W) -> defaulting to 0.0
            inputs_dict["latlons"] = torch.zeros((batch_size, 2, H, W), dtype=torch.float32, device=x.device)

            # Inject the required keyword arguments for Presto
            kwargs = {
                "inputs": inputs_dict
            }
            
            return None, kwargs
            
        else:
            raise ValueError(f"PrestoAdapter expects a torch.Tensor, but got {type(x)}.")