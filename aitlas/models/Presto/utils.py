from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import torch


# Constants 
PRESTO_S1_BANDS = ["vv", "vh"]
_PRESTO_S2_BANDS_ALL = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
]
ERA5_BANDS = ["temperature_2m", "total_precipitation"]
SRTM_BANDS = ["elevation", "slope"]
_PRESTO_BANDS_ALL = PRESTO_S1_BANDS + _PRESTO_S2_BANDS_ALL + ERA5_BANDS + SRTM_BANDS + ["NDVI"]

# The actual bands used by the model (B09 is excluded)
INPUT_PRESTO_S2_BANDS = [b for b in _PRESTO_S2_BANDS_ALL if b != "B09"]
INPUT_PRESTO_BANDS = [b for b in _PRESTO_BANDS_ALL if b != "B09"]
NUM_DYNAMIC_WORLD_CLASSES = 9

# For normalization
PRESTO_ADD_BY = torch.Tensor(
    [
        25.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -272.15, 0.0, 0.0, 0.0, 0.0,
    ]
)
PRESTO_DIV_BY = torch.Tensor(
    [
        25.0, 25.0, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4,
        35.0, 0.03, 2000.0, 50.0, 1.0,
    ]
)

BANDS_GROUPS_IDX = OrderedDict(
    [
        ("S1", [0, 1]),
        ("S2_RGB", [2, 3, 4]),
        ("S2_Red_Edge", [5, 6, 7]),
        ("S2_NIR_10m", [8]),
        ("S2_NIR_20m", [9]),
        ("S2_SWIR", [10, 11]),
        ("ERA5", [12, 13]),
        ("SRTM", [14, 15]),
        ("NDVI", [16]),
    ]
)

def prepare_presto_input(
    s1: Optional[torch.Tensor] = None,
    s2: Optional[torch.Tensor] = None,
    era5: Optional[torch.Tensor] = None,
    srtm: Optional[torch.Tensor] = None,
    dynamic_world: Optional[torch.Tensor] = None,
    latlons: Optional[torch.Tensor] = None,
    months: Optional[torch.Tensor] = None,
    s2_bands: List[str] = INPUT_PRESTO_S2_BANDS,
    default_month: int = 6,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Handles all data preparation for the Presto wrapper.

    Takes raw batched tensors and returns a complete, consistent set of processed
    tensors (x, dynamic_world, latlons, months) in a spatial format.
    """

    data_sources = [s for s in [s1, s2, era5, srtm] if s is not None]
    if not data_sources:
        raise ValueError("At least one data source (s1, s2, etc.) must be provided.")

    # Validate shapes
    b, t, h, w, device = (
        data_sources[0].shape[0],
        data_sources[0].shape[1],
        data_sources[0].shape[3],
        data_sources[0].shape[4],
        data_sources[0].device,
    )

    # Prepare the main `x` data tensor
    x = torch.zeros(b, t, len(INPUT_PRESTO_BANDS), h, w, device=device)

    band_mappings = [
        (s1, PRESTO_S1_BANDS), (s2, s2_bands), (era5, ERA5_BANDS), (srtm, SRTM_BANDS)
    ]
    for data, input_bands in band_mappings:
        if data is not None:
            output_indices = [INPUT_PRESTO_BANDS.index(band) for band in input_bands]
            x[:, :, output_indices, :, :] = data

    # Calculate and insert NDVI (if s2 is provided)
    if s2 is not None:
        nir_band, red_band = "B08", "B04"
        if nir_band in s2_bands and red_band in s2_bands:
            nir_idx, red_idx = s2_bands.index(nir_band), s2_bands.index(red_band)
            nir, red = s2[:, :, nir_idx, :, :], s2[:, :, red_idx, :, :]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi_output_idx = INPUT_PRESTO_BANDS.index("NDVI")
            x[:, :, ndvi_output_idx, :, :] = ndvi

    # Normalize `x`
    if normalize:
        add_by = PRESTO_ADD_BY.view(1, 1, -1, 1, 1).to(device)
        div_by = PRESTO_DIV_BY.view(1, 1, -1, 1, 1).to(device)
        x = (x + add_by) / div_by

    # Prepare `dynamic_world`
    if dynamic_world is None:
        dynamic_world = torch.full((b, t, h, w), NUM_DYNAMIC_WORLD_CLASSES, device=device)

    # Prepare `months`
    final_months: torch.Tensor
    if months is None:
        # Create sequence from default start month for the whole batch
        month_sequence = torch.fmod(torch.arange(default_month, default_month + t, dtype=torch.long), 12)
        final_months = month_sequence.expand(b, t).to(device)
    elif len(months.shape) == 1: # (B,) of start months
        # Create a unique sequence for each sample in the batch
        final_months = torch.stack([
            torch.fmod(torch.arange(m, m + t, dtype=torch.long), 12) for m in months
        ]).to(device)
    else: # Assumes months is already (B, T)
        final_months = months.to(device)

    # Pass latlons through without modification
    if latlons is None:
        # Presto encoder requires latlons, raise error or create a default
        raise ValueError("`latlons` tensor is required but was not provided.")

    return x, dynamic_world.long(), latlons, final_months.long()