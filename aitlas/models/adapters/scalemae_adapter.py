import torch

from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("ScaleMAEAdapter")
class ScaleMAEAdapter(BaseInputAdapter):
    def forward(self, x: torch.Tensor):
        kwargs = {}

        if isinstance(x, torch.Tensor):

            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Extract the bands from the user's config
            bands_list = getattr(self.config, "bands", [])

            # Scale-MAE is strictly an RGB model
            allowed_bands = ["B02", "B03", "B04"]

            # Default to RGB bands if not provided or empty
            if not bands_list:
                bands_list = allowed_bands.copy()

            # Enforce that the provided bands exactly match the required RGB bands
            if set(bands_list) != set(allowed_bands):
                raise ValueError(
                    f"Scale-MAE models require exactly the RGB bands: "
                    f"{allowed_bands}. You provided: {bands_list}"
                )

            # Handle the ground sample distance (GSD) input as input_res parameter in forward_params from the config
            # Get the batch size from the input tensor
            batch_size = x.shape[0]

            # Extract the raw forward_params from the config (fallback to empty dict)
            forward_params = getattr(self.config, "forward_params", {}) or {}

            # Get the resolution value the user specified (defaulting to 10.0m for Sentinel-2)
            input_res = forward_params.get("input_res", 10.0)

            # Upgrade the raw float into a (B,) tensor on the correct device
            input_res_tensor = torch.full(
                (batch_size,), float(input_res), device=x.device, dtype=x.dtype
            )

            # Inject the required keyword arguments for ScaleMAE
            kwargs["input_res"] = input_res_tensor

        return x, kwargs
