import torch
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("TerraMindAdapter")
class TerraMindAdapter(BaseInputAdapter):
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            
            # Replace NaNs and Infs
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Fetch the modalities list from the config safely
            modalities = getattr(self.config, "modalities", None) or []

            # Dynamically determine the correct keys based on the modalities list
            s1_key = next((m for m in modalities if m in ["S1GRD", "S1RTC"]), None)
            s2_key = next((m for m in modalities if m in ["S2L2A", "S2L1C"]), None)
            dem_key = next((m for m in modalities if m == "DEM"), None)

            # Define default fallback lists based on TerraMind's PRETRAINED_BANDS
            default_s2 = []
            if s2_key == "S2L2A":
                default_s2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
            elif s2_key == "S2L1C":
                default_s2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

            # Get bands from the config safely, or fallback to defaults if the modality is active
            bands_s1 = getattr(self.config, "bands_s1", None) or (["VV", "VH"] if s1_key else [])
            bands_s2 = getattr(self.config, "bands_s2", None) or default_s2
            bands_dem = getattr(self.config, "bands_dem", None) or (["DEM"] if dem_key else [])

            # If input is just RGB images
            is_pure_rgb = (modalities == ["RGB"]) or (
                not any([bands_s1, bands_s2, bands_dem]) and x.shape[1] == 3
            )
            
            if is_pure_rgb:
                return {"RGB": x}, {}

            inputs_dict = {}
            current_idx = 0
            
            # Slice dynamically based on the number of bands
            if bands_s1 and s1_key:
                num_s1 = len(bands_s1)
                inputs_dict[s1_key] = x[:, current_idx : current_idx + num_s1, ...]
                current_idx += num_s1
                
            if bands_s2 and s2_key:
                num_s2 = len(bands_s2)
                inputs_dict[s2_key] = x[:, current_idx : current_idx + num_s2, ...]
                current_idx += num_s2

            if bands_dem and dem_key:
                num_dem = len(bands_dem)
                inputs_dict[dem_key] = x[:, current_idx : current_idx + num_dem, ...]
                current_idx += num_dem

            return inputs_dict, {}