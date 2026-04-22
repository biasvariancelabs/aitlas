import torch

from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("SatMAEAdapter")
class SatMAEAdapter(BaseInputAdapter):
    def forward(self, x: torch.Tensor):
        kwargs = {}

        if isinstance(x, torch.Tensor):

            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Extract the bands and backbone name from the user's config
            bands_list = getattr(self.config, "bands", [])

            # Use getattr with a fallback to "" to safely handle cases where it isn't defined
            backbone_name = getattr(self.config, "backbone_name", "")
            backbone_name = str(backbone_name).lower()

            # SatMAE multispectral expects exactly 10 specific Sentinel-2 bands
            allowed_bands = [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
            ]

            # Backbone validation logic
            # Multispectral backbone
            if "multispectral" in backbone_name:
                # Default to all 10 bands if missing
                if not bands_list:
                    bands_list = allowed_bands.copy()

                # Enforce that the provided bands exactly match the required 10 bands
                if set(bands_list) != set(allowed_bands):
                    raise ValueError(
                        f"SatMAE multispectral models require exactly these 10 bands in entirety: "
                        f"{allowed_bands}. You provided: {bands_list}"
                    )
            # Standard RGB backbone (not multispectral, not temporal)
            elif "temporal" not in backbone_name:
                rgb_bands = ["B02", "B03", "B04"]

                # Default to RGB if missing
                if not bands_list:
                    bands_list = rgb_bands

                # Enforce that only RGB bands are used
                if set(bands_list) != set(rgb_bands):
                    raise ValueError(
                        f"SatMAE standard (RGB) models require exactly the RGB bands. "
                        f"Expected: {rgb_bands}, but found: {bands_list}"
                    )

            else:
                # Temporal mode (not implemented)
                raise NotImplementedError(
                    "SatMAE temporal mode is currently not implemented. "
                    "Please use a standard RGB or multispectral backbone instead."
                )

        return x, kwargs
