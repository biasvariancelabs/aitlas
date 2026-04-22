# Sentinel-2 data
# Central wavelengths (in micrometers, µm)
SENTINEL_2_WAVELENGTHS = {
    "B01": 0.443,
    "B02": 0.490,
    "B03": 0.560,
    "B04": 0.665,
    "B05": 0.705,
    "B06": 0.740,
    "B07": 0.783,
    "B08": 0.842,
    "B8A": 0.865,
    "B09": 0.945,
    "B10": 1.375,
    "B11": 1.610,
    "B12": 2.190,
}

# Bandwidths (in micrometers, µm)
SENTINEL_2_BANDWIDTHS = {
    "B01": 0.020,
    "B02": 0.065,
    "B03": 0.035,
    "B04": 0.030,
    "B05": 0.015,
    "B06": 0.015,
    "B07": 0.020,
    "B08": 0.115,
    "B8A": 0.020,
    "B09": 0.020,
    "B10": 0.030,
    "B11": 0.090,
    "B12": 0.180,
}

# Standard 13-band stack indices (0-indexed)
SENTINEL_2_BAND_INDICES = {
    "B01": 0,
    "B02": 1,
    "B03": 2,
    "B04": 3,
    "B05": 4,
    "B06": 5,
    "B07": 6,
    "B08": 7,
    "B8A": 8,
    "B09": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}

# Spatial resolutions (in meters)
SENTINEL_2_RESOLUTIONS = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
}


# Sentinel-1 data
SENTINEL_1_WAVELENGTH = [0.05546, 0.05546]
SENTINEL_1_FREQUENCY_GHZ = [5.405, 5.405]

# Sentinel-1 polarization channels
SENTINEL_1_POLARIZATIONS = {"VV": 0, "VH": 1, "HH": 2, "HV": 3}

# Sentinel-1 RTC for pseudo-RGB visualization and features
SENTINEL_1_RTC_INDICES = {
    "VV": 0,
    "VH": 1,
    "VV/VH": 2,  # Common derived channel for pseudo-RGB visualization and features
}

# Made-up wavelengths for Sentinel-1 channels (used in DOFA, etc.)
SENTINEL_1_WAVELENGTHS_MADE_UP = [3.75, 3.75]

# Sentinel-1 channel IDs for Panopticon
SENTINEL_1_IDS = {"VV": -1, "VH": -2, "HH": -3, "HV": -4}


# Landsat 8 data
LANDSAT_8_WAVELENGTHS = {
    "B01": 0.443,
    "B02": 0.490,
    "B03": 0.560,
    "B04": 0.665,
    "B05": 0.705,
    "B06": 0.740,
    "B07": 0.783,
    "B08": 0.842,
    "B09": 0.945,
    "B10": 1.375,
    "B11": 1.610,
}
