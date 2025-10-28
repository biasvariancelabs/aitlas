from enum import Enum
from typing import Any

class HLSBands(Enum):
    COASTAL_AEROSOL = "COASTAL_AEROSOL"
    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    RED_EDGE_1 = "RED_EDGE_1"
    RED_EDGE_2 = "RED_EDGE_2"
    RED_EDGE_3 = "RED_EDGE_3"
    NIR_BROAD = "NIR_BROAD"
    NIR_NARROW = "NIR_NARROW"
    SWIR_1 = "SWIR_1"
    SWIR_2 = "SWIR_2"
    WATER_VAPOR = "WATER_VAPOR"
    CIRRUS = "CIRRUS"
    THERMAL_INFRARED_1 = "THERMAL_INFRARED_1"
    THERMAL_INFRARED_2 = "THERMAL_INFRARED_2"

    @classmethod
    def try_convert_to_hls_bands_enum(cls, x: Any):
        try:
            return cls(x)
        except ValueError:
            return x

PRETRAINED_BANDS = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]

PRITHVI_V1_MEAN = [775.0, 1081.0, 1229.0, 2497.0, 2204.0, 1611.0]
PRITHVI_V1_STD = [1282.0, 1270.0, 1399.0, 1368.0, 1292.0, 1155.0]
PRITHVI_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]