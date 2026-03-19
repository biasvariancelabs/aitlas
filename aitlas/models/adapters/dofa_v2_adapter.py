import torch
from aitlas.base.adapters import BaseInputAdapter
from aitlas.models.registries import ADAPTER_REGISTRY
from .utils import SENTINEL_2_WAVELENGTHS, SENTINEL_2_BANDWIDTHS


@ADAPTER_REGISTRY.register("DOFAAdapter")
class DOFAAdapter(BaseInputAdapter):
    def forward(self, x: torch.Tensor):
        kwargs = {}
        
        if isinstance(x, torch.Tensor):
            
            # Replace NaNs and Infs with 0.0 to prevent issues with model training
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Extract the bands from the user's config
            bands_list = getattr(self.config, "bands", [])
            
            # Default to RGB bands if not provided or empty
            if not bands_list:
                bands_list = ["B02", "B03", "B04"]
                
            # Define the exact set of bands this model is allowed to process
            allowed_bands = [
                "B01", "B02", "B03", "B04", "B05", "B06", 
                "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
            ]
            
            # Prepare wavelengths list
            # Map bands to wavelengths (in microns)
            wave_list = []
                
            for band in bands_list:
                # Check if the band can be input into DOFA
                if band not in allowed_bands:
                    raise ValueError(
                        f"Invalid band '{band}' found in config. "
                        f"DOFA allows the following bands: {allowed_bands}"
                    )
                
                # Extract the wavelengths
                wave_list.append(float(SENTINEL_2_WAVELENGTHS[band]))
            
            # Inject the required keyword arguments for DOFA
            kwargs["wave_list"] = wave_list
            
        return x, kwargs