from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange

# Constants
PRESTO_S1_BANDS = ["vv", "vh"]
_PRESTO_S2_BANDS_ALL = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
]
ERA5_BANDS = ["temperature_2m", "total_precipitation"]
SRTM_BANDS = ["elevation", "slope"]
_PRESTO_BANDS_ALL = _PRESTO_S2_BANDS_ALL + PRESTO_S1_BANDS + ERA5_BANDS + SRTM_BANDS + ["NDVI"]

# The actual bands used by the model (B09 is excluded)
INPUT_PRESTO_S2_BANDS = [b for b in _PRESTO_S2_BANDS_ALL if b != "B09"]
INPUT_PRESTO_BANDS = [b for b in _PRESTO_BANDS_ALL if b != "B09"]
NUM_DYNAMIC_WORLD_CLASSES = 9

# For normalization
PRESTO_ADD_BY = torch.Tensor(
    # S1, S2 (10 bands), ERA5, SRTM, NDVI
    [
        25.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -272.15,
        0.0, 0.0, 0.0, 0.0,
    ]
)
PRESTO_DIV_BY = torch.Tensor(
    [
        25.0, 25.0, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 35.0,
        0.03, 2000.0, 50.0, 1.0,
    ]
)

def prepare_presto_input(
    s1: Optional[torch.Tensor] = None,
    s2: Optional[torch.Tensor] = None,
    era5: Optional[torch.Tensor] = None,
    srtm: Optional[torch.Tensor] = None,
    dynamic_world: Optional[torch.Tensor] = None,
    months: Optional[torch.Tensor] = None,
    s1_bands: List[str] = PRESTO_S1_BANDS,
    s2_bands: List[str] = INPUT_PRESTO_S2_BANDS,
    era5_bands: List[str] = ERA5_BANDS,
    srtm_bands: List[str] = SRTM_BANDS,
    default_month: int = 6,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs and normalizes the input tensors for the Presto model from individual data sources.
    This function is a standalone version of the data preparation logic.
    """
    # Input validation
    sources = [s1, s2, era5, srtm]
    sources_present = [x for x in sources if x is not None]
    if not sources_present:
        raise ValueError("At least one data source (s1, s2, era5, srtm) must be provided.")

    shapes = {
        "batch_size": [d.shape[0] for d in sources_present],
        "height": [d.shape[2] for d in sources_present],
        "width": [d.shape[3] for d in sources_present],
        "device": [d.device for d in sources_present],
    }
    for name, values in shapes.items():
        if len(set(values)) != 1:
            raise ValueError(f"Inconsistent {name} across input tensors.")
    b, h, w, device = shapes["batch_size"][0], shapes["height"][0], shapes["width"][0], shapes["device"][0]

    # Construct the main input tensors
    x: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    t: Optional[int] = None

    for data, input_bands in [(s1, s1_bands), (s2, s2_bands), (era5, era5_bands), (srtm, srtm_bands)]:
        if data is None:
            continue
        
        num_bands = len(input_bands)
        cur_t = data.shape[1] // num_bands
        if t is None:
            t = cur_t
        elif t != cur_t:
            raise ValueError(f"Inconsistent number of timesteps found: {t} and {cur_t}")

        data_reshaped = rearrange(data, "b (t c) h w -> b t h w c", t=t, c=num_bands)

        if x is None:
            x = torch.zeros(b, t, h, w, len(INPUT_PRESTO_BANDS), device=device)
            mask = torch.ones(b, t, h, w, len(INPUT_PRESTO_BANDS), device=device)

        output_indices = [INPUT_PRESTO_BANDS.index(band) for band in input_bands]
        x[:, :, :, :, output_indices] = data_reshaped
        mask[:, :, :, :, output_indices] = 0

    if x is None or mask is None or t is None:
         raise ValueError("Failed to construct input tensors. Check provided data sources.")

    # Handle "dynamic_world" and "months"
    if dynamic_world is None:
        dynamic_world = torch.full((b, t, h, w), NUM_DYNAMIC_WORLD_CLASSES, device=device)

    if months is None:
        months = torch.full((b, t), default_month, device=device)
    else:
        # If months are provided per sample, they might be (B,) instead of (B, T)
        if len(months.shape) == 1:
            months = months.unsqueeze(1).expand(-1, t)

    # Normalize data
    if normalize:
        add_by = PRESTO_ADD_BY.view(1, 1, 1, 1, -1).to(device)
        div_by = PRESTO_DIV_BY.view(1, 1, 1, 1, -1).to(device)
        x = (x + add_by) / div_by

    return x, mask, dynamic_world.long(), months.long()