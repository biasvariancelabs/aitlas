import torch
from torchvision.transforms import v2
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("CROMAAdapter")
class CROMAAdapter(BaseInputAdapter):
    def forward(self, x):
        kwargs = {}
        modalities = []
        
        # Check the input type
        if isinstance(x, torch.Tensor):
            # Clean the single tensor
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            x_dict = {}
            
            # Check the config for S1 and S2 bands
            has_s1 = hasattr(self.config, "bands_s1")
            has_s2 = hasattr(self.config, "bands_s2")
            
            # Fetch the band lists (fallback to standard defaults if empty or not present)
            bands_s1 = getattr(self.config, "bands_s1", []) or ["VV", "VH"]
            bands_s2 = getattr(self.config, "bands_s2", []) or [
                "B01", "B02", "B03", "B04", "B05", "B06", 
                "B07", "B08", "B8A", "B09", "B11", "B12"
            ]
            
            # Calculate dynamic slice sizes directly from config
            num_s1 = len(bands_s1)
            num_s2 = len(bands_s2)
            actual_channels = x.shape[1] if x.dim() >= 2 else 0
            
            if has_s1 and has_s2 and actual_channels >= (num_s1 + num_s2):
                # Multimodal Stacked: S1 is always first, S2 is always second
                x_dict["sentinel_1"] = x[:, :num_s1, ...]
                x_dict["sentinel_2"] = x[:, num_s1 : num_s1 + num_s2, ...]
                
            elif has_s1 and not has_s2 and actual_channels >= num_s1:
                # Sentinel-1 Only
                x_dict["sentinel_1"] = x[:, :num_s1, ...]
                
            elif has_s2 and not has_s1 and actual_channels >= num_s2:
                # Sentinel-2 Only
                x_dict["sentinel_2"] = x[:, :num_s2, ...]
                
            else:
                expected_channels = (num_s1 + num_s2) if (has_s1 and has_s2) else (num_s1 if has_s1 else num_s2)
                raise ValueError(
                    f"Config expects {'S1 + S2' if has_s1 and has_s2 else 'S1' if has_s1 else 'S2'} "
                    f"(expected {expected_channels} channels), but the input tensor has {actual_channels} channels."
                )

        # Sentinel-1 (SAR) data handling
        if "sentinel_1" in x_dict and x_dict["sentinel_1"] is not None:
            x_sar = x_dict["sentinel_1"]
            
            # Enforce spatial dimensions (120x120)
            if x_sar.shape[-2:] != (120, 120):
                data_transforms = v2.Resize((120, 120), antialias=True)
                x_sar = data_transforms(x_sar)
            
            # Enforce exactly VV and VH bands
            bands_s1 = getattr(self.config, "bands_s1", []) or ["VV", "VH"]
            allowed_s1 = ["VV", "VH"]
            if set(bands_s1) != set(allowed_s1):
                raise ValueError(
                    f"CROMA requires exactly these Sentinel-1 bands: {allowed_s1}. "
                    f"You provided: {bands_s1}"
                )
            
            kwargs["x_sar"] = x_sar
            modalities.append("sar")
            
        # Sentinel-2 (optical) data handling
        if "sentinel_2" in x_dict and x_dict["sentinel_2"] is not None:
            x_opt = x_dict["sentinel_2"]
            
            # Enforce spatial dimensions (120x120)
            if x_opt.shape[-2:] != (120, 120):
                data_transforms = v2.Resize((120, 120), antialias=True)
                x_opt = data_transforms(x_opt)
            
            # Enforce exactly the 12 CROMA bands
            bands_s2 = getattr(self.config, "bands_s2", []) or [
                "B01", "B02", "B03", "B04", "B05", "B06", 
                "B07", "B08", "B8A", "B09", "B11", "B12"
            ]
            allowed_s2 = [
                "B01", "B02", "B03", "B04", "B05", "B06", 
                "B07", "B08", "B8A", "B09", "B11", "B12"
            ]
            if set(bands_s2) != set(allowed_s2):
                raise ValueError(
                    f"CROMA requires exactly 12 Sentinel-2 bands: {allowed_s2}. "
                    f"You provided: {bands_s2}"
                )
            
            kwargs["x_optical"] = x_opt
            modalities.append("optical")
            
        # Ensure at least one modality was successfully routed
        if not modalities:
            raise ValueError("CROMA requires at least one valid modality to process.")
            
        # Inject the active modalities list
        kwargs["modalities"] = modalities
        
        return None, kwargs