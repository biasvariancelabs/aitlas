import torch
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY
from .utils import SENTINEL_2_WAVELENGTHS, SENTINEL_2_BANDWIDTHS


@ADAPTER_REGISTRY.register("CopernicusFMAdapter")
class CopernicusFMAdapter(BaseInputAdapter):
    def forward(self, x: torch.Tensor):
        kwargs = {}
        
        if isinstance(x, torch.Tensor):
            
            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Get batch size from the input tensor
            batch_size = x.shape[0]
            
            # Prepare metadata: [longitude, latitude, timestamp, patch_area]
            # Since the dataloader only passes the image, we safely bypass this with NaNs
            dummy_metadata = torch.full(
                (batch_size, 4), 
                float('nan'), 
                device=x.device, 
                dtype=x.dtype
            )
            kwargs["metadata"] = dummy_metadata
        
        # Prepare wavelengths and bandwidths
        bands_list = getattr(self.config, "bands", [])
        if bands_list:
            # Convert from micrometers to nanometers (* 1000)
            wavelengths = [
                float(1000 * SENTINEL_2_WAVELENGTHS.get(band, 0.0)) 
                for band in bands_list
            ]
            bandwidths = [
                float(1000 * SENTINEL_2_BANDWIDTHS.get(band, 0.0)) 
                for band in bands_list
            ]
            
            kwargs["wavelengths"] = wavelengths
            kwargs["bandwidths"] = bandwidths
            kwargs["input_mode"] = "spectral"
            
        return x, kwargs