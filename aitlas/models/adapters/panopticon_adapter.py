import torch

from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY

from .utils import LANDSAT_8_WAVELENGTHS, SENTINEL_1_IDS, SENTINEL_2_WAVELENGTHS


@ADAPTER_REGISTRY.register("PanopticonAdapter")
class PanopticonAdapter(BaseInputAdapter):
    def forward(self, x):
        kwargs = {}

        if isinstance(x, torch.Tensor):
            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Check the config for S1, S2 and L8 bands
            has_s1 = hasattr(self.config, "bands_s1")
            has_s2 = hasattr(self.config, "bands_s2")
            has_l8 = hasattr(self.config, "bands_l8")

            # Fetch the band lists (fallback to standard defaults if empty or not present)
            bands_s1 = getattr(self.config, "bands_s1", []) or ["VV", "VH"]
            bands_s2 = getattr(self.config, "bands_s2", []) or [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B09",
                "B11",
                "B12",
            ]
            bands_l8 = getattr(self.config, "bands_l8", []) or [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B09",
                "B10",
                "B11",
            ]

            # Calculate expected slice sizes
            num_s1 = len(bands_s1) if has_s1 else 0
            num_s2 = len(bands_s2) if has_s2 else 0
            num_l8 = len(bands_l8) if has_l8 else 0
            expected_channels = num_s1 + num_s2 + num_l8
            actual_channels = x.shape[1] if x.dim() >= 2 else 0

            if actual_channels < expected_channels:
                raise ValueError(
                    f"Config expects {expected_channels} channels "
                    f"but the input tensor only has {actual_channels} channels."
                )

            # Slice off any extra "garbage" channels attached to the end of the tensor
            x = x[:, :expected_channels, ...]

            # Generate chn_ids tensor
            chn_list = []

            # We append S1 first, then S2 and L8 bands
            if has_s1:
                for band in bands_s1:
                    if band not in SENTINEL_1_IDS:
                        raise ValueError(
                            f"Invalid SAR band {band}. Must be one of {list(SENTINEL_1_IDS.keys())}."
                        )
                    chn_list.append(SENTINEL_1_IDS[band])

            if has_s2:
                for band in bands_s2:
                    if band not in SENTINEL_2_WAVELENGTHS:
                        raise ValueError(
                            f"Invalid Optical band {band}. Must be one of {list(SENTINEL_2_WAVELENGTHS.keys())}."
                        )
                    chn_list.append(float(1000 * SENTINEL_2_WAVELENGTHS[band]))

            if has_l8:
                for band in bands_l8:
                    if band not in LANDSAT_8_WAVELENGTHS:
                        raise ValueError(
                            f"Invalid Optical band {band}. Must be one of {list(LANDSAT_8_WAVELENGTHS.keys())}."
                        )
                    chn_list.append(float(1000 * LANDSAT_8_WAVELENGTHS[band]))

            # Calculate the actual batch size
            batch_size = x.shape[0] if x.dim() == 4 else 1

            # Convert chn_ids list into a tensor of shape (B, C)
            # We repeat the 1D list across the batch dimension
            chn_ids = torch.tensor(chn_list, dtype=torch.float32, device=x.device)
            chn_ids = chn_ids.unsqueeze(0).repeat(batch_size, 1)

            # Inject the required keyword arguments for Panopticon
            kwargs["x_dict"] = {"imgs": x, "chn_ids": chn_ids}

        # Return None as the primary 'x' to force kwargs routing in CompositeModel
        return None, kwargs
