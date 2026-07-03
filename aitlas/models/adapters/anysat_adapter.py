import torch

from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY


@ADAPTER_REGISTRY.register("AnySatAdapter")
class AnySatAdapter(BaseInputAdapter):
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            # Replace NaNs and Infs
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Get actual batch size safely
            batch_size = x.shape[0]

            # Galileo expects exactly 2 S1 bands (VV, VH) and 10 S2 bands (no B1, B9, B10), and 11 L8 bands
            ANYSAT_S1_BANDS = ["VV", "VH"]
            ANYSAT_S2_BANDS = [
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
            ANYSAT_L8_BANDS = [
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

            # Check config flags
            has_s1 = hasattr(self.config, "bands_s1")
            has_s2 = hasattr(self.config, "bands_s2")
            has_l8 = hasattr(self.config, "bands_l8")

            # Fetch the band lists (fallback to standard defaults if empty)
            bands_s1 = getattr(self.config, "bands_s1", []) or ANYSAT_S1_BANDS
            bands_s2 = getattr(self.config, "bands_s2", []) or ANYSAT_S2_BANDS
            bands_l8 = getattr(self.config, "bands_l8", []) or ANYSAT_L8_BANDS

            # Ensure config matches AnySat's requirements
            if has_s1 and set(bands_s1) != set(ANYSAT_S1_BANDS):
                raise ValueError(
                    f"AnySat requires exactly these S1 bands: {ANYSAT_S1_BANDS}. Got: {bands_s1}"
                )
            if has_s2 and set(bands_s2) != set(ANYSAT_S2_BANDS):
                raise ValueError(
                    f"AnySat requires exactly these S2 bands: {ANYSAT_S2_BANDS}. Got: {bands_s2}"
                )
            if has_l8 and set(bands_l8) != set(ANYSAT_L8_BANDS):
                raise ValueError(
                    f"AnySat requires exactly these L8 bands: {ANYSAT_L8_BANDS}. Got: {bands_l8}"
                )

            # Calculate slice sizes
            num_s1 = len(bands_s1) if has_s1 else 0
            num_s2 = len(bands_s2) if has_s2 else 0
            num_l8 = len(bands_l8) if has_l8 else 0

            expected_channels = num_s1 + num_s2 + num_l8
            actual_channels = x.shape[1]

            if actual_channels < expected_channels:
                raise ValueError(
                    f"Config expects {expected_channels} channels "
                    f"but the input tensor only has {actual_channels} channels."
                )

            # Slice modalities
            x_dict = {}
            current_idx = 0

            if has_s1:
                # Slice S1 -> Shape: (B, 2, H, W)
                s1_tensor = x[:, current_idx : current_idx + num_s1, ...]
                # Generate the 3rd channel (VV/VH)
                epsilon = 1e-8  # Add a tiny epsilon to prevent division by zero
                ratio_channel = s1_tensor[:, 0, ...] / (s1_tensor[:, 1, ...] + epsilon)
                # Normalize the new ratio channel to [0, 1]
                r_min = ratio_channel.amin(dim=(-2, -1), keepdim=True)
                r_max = ratio_channel.amax(dim=(-2, -1), keepdim=True)
                ratio_channel = (ratio_channel - r_min) / (r_max - r_min + epsilon)
                ratio_channel = ratio_channel.unsqueeze(1)  # Shape: (B, 1, H, W)
                # Concatenate to create 3 channels: (B, 3, H, W)
                s1_3d = torch.cat((s1_tensor, ratio_channel), dim=1)
                # Unsqueeze temporal dim -> Shape: (B, 1, 3, H, W)
                x_dict["s1"] = s1_3d.unsqueeze(1)
                # Complementary dates tensor -> Shape: (B, 1)
                x_dict["s1_dates"] = torch.zeros(
                    (batch_size, 1), dtype=torch.float32, device=x.device
                )
                current_idx += num_s1

            if has_s2:
                # Slice -> unsqueeze temporal dim -> (B, 1, 10, H, W), where T=1
                x_dict["s2"] = x[:, current_idx : current_idx + num_s2, ...].unsqueeze(1)
                # Complementary dates tensor -> Shape: (B, 1)
                x_dict["s2_dates"] = torch.zeros(
                    (batch_size, 1), dtype=torch.float32, device=x.device
                )
                current_idx += num_s2

            if has_l8:
                # Slice -> Unsqueeze temporal dim -> (B, 1, 11, H, W), where T=1
                x_dict["l8"] = x[:, current_idx : current_idx + num_l8, ...].unsqueeze(1)
                # Complementary dates tensor -> Shape: (B, 1)
                x_dict["l8_dates"] = torch.zeros(
                    (batch_size, 1), dtype=torch.float32, device=x.device
                )

            # Inject the required keyword arguments for AnySat
            kwargs = {"x": x_dict}

            # Return None as the primary 'x' to force kwargs routing
            return None, kwargs
