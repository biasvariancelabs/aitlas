import torch

from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("CACoAdapter")
class CACoAdapter(BaseInputAdapter):
    def forward(self, x: torch.Tensor):
        kwargs = {}

        if isinstance(x, torch.Tensor):
            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Extract the bands from the user's config
            bands_list = getattr(self.config, "bands", [])

            # CACo is strictly an RGB model
            allowed_bands = ["B02", "B03", "B04"]

            # Default to RGB bands if not provided or empty
            if not bands_list:
                bands_list = allowed_bands.copy()

            # Enforce that the provided bands exactly match the required RGB bands
            if set(bands_list) != set(allowed_bands):
                raise ValueError(
                    f"CACo models require exactly the RGB bands: "
                    f"{allowed_bands}. You provided: {bands_list}"
                )

        return x, kwargs
