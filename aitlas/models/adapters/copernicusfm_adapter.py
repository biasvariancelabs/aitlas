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
        
        # Default to RGB bands if not provided or empty
        if not bands_list:
            bands_list = ["B02", "B03", "B04"]
            
        # Define the exact set of bands this model is allowed to process
        allowed_bands = [
            "B01", "B02", "B03", "B04", "B05", "B06", 
            "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
        ]
            
        # Prepare wavelengths and bandwidths
        # Convert from micrometers to nanometers (* 1000)
        wavelengths = []
        bandwidths = []
        
        for band in bands_list:
            # Check if the band can be input into CopernicusFM
            if band not in allowed_bands:
                raise ValueError(
                    f"Invalid band '{band}' found in config. "
                    f"CopernicusFM allows the following bands: {allowed_bands}"
                )
            
            # Extract the wavelengths and bandwidths (Guaranteed to exist!)
            wavelengths.append(float(1000 * SENTINEL_2_WAVELENGTHS[band]))
            bandwidths.append(float(1000 * SENTINEL_2_BANDWIDTHS[band]))
        
        # Inject the required keyword arguments for CopernicusFM
        kwargs["wavelengths"] = wavelengths
        kwargs["bandwidths"] = bandwidths
        kwargs["input_mode"] = "spectral"
            
        return x, kwargs