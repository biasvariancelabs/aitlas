import math
from copy import deepcopy
from typing import NamedTuple, OrderedDict, cast
from typing import OrderedDict as OrderedDictType

import numpy as np
import torch


# constants
CONFIG_FILENAME = "config.json"
ENCODER_FILENAME = "encoder.pt"
BASE_GSD = 10
DEFAULT_MONTH = 5

# band information
S1_BANDS = ["VV", "VH"]
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
S2_SHIFT_VALUES = [0.0] * len(S2_BANDS)
S2_DIV_VALUES = [1e4] * len(S2_BANDS)
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
# for temperature, shift to celcius and then divide by 35 based on notebook (ranges from)
# 37 to -22 degrees celcius
# For rainfall, based on
# https://github.com/nasaharvest/presto/blob/main/notebooks/exploratory_data_analysis.ipynb
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]
TC_BANDS = ["def", "soil", "aet"]
TC_SHIFT_VALUES = [0.0, 0.0, 0.0]
TC_DIV_VALUES = [4548, 8882, 2000]
VIIRS_BANDS = ["avg_rad"]
VIIRS_SHIFT_VALUES = [0.0]
# visually checked - this seems much more reasonable than
# the GEE estimate
VIIRS_DIV_VALUES = [100]
SRTM_BANDS = ["elevation", "slope"]
# visually gauged 90th percentile from
# https://github.com/nasaharvest/presto/blob/main/notebooks/exploratory_data_analysis.ipynb
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]
DW_BANDS = [
    "DW_water",
    "DW_trees",
    "DW_grass",
    "DW_flooded_vegetation",
    "DW_crops",
    "DW_shrub_and_scrub",
    "DW_built",
    "DW_bare",
    "DW_snow_and_ice",
]
DW_SHIFT_VALUES = [0] * len(DW_BANDS)
DW_DIV_VALUES = [1] * len(DW_BANDS)

WC_BANDS = [
    "WC_temporarycrops",
    "WC_maize",
    "WC_wintercereals",
    "WC_springcereals",
    "WC_irrigation",
]
WC_SHIFT_VALUES = [0] * len(WC_BANDS)
WC_DIV_VALUES = [100] * len(WC_BANDS)
STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]

LANDSCAN_BANDS = ["b1"]
# LANDSCAN values range from approximately 0 to 185000 in 2022: https://code.earthengine.google.com/?scriptPath=users/sat-io/awesome-gee-catalog-examples:population-socioeconomics/LANDSCAN-GLOBAL
LANDSCAN_SHIFT_VALUES = [92500]
LANDSCAN_DIV_VALUES = [92500]
LOCATION_BANDS = ["x", "y", "z"]

SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]
TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS

# 0 for NDVI
SPACE_TIME_SHIFT_VALUES = np.array(S1_SHIFT_VALUES + S2_SHIFT_VALUES + [0])
SPACE_TIME_DIV_VALUES = np.array(S1_DIV_VALUES + S2_DIV_VALUES + [1])
TIME_SHIFT_VALUES = np.array(ERA5_SHIFT_VALUES + TC_SHIFT_VALUES + VIIRS_SHIFT_VALUES)
TIME_DIV_VALUES = np.array(ERA5_DIV_VALUES + TC_DIV_VALUES + VIIRS_DIV_VALUES)
SPACE_SHIFT_VALUES = np.array(SRTM_SHIFT_VALUES + DW_SHIFT_VALUES + WC_SHIFT_VALUES)
SPACE_DIV_VALUES = np.array(SRTM_DIV_VALUES + DW_DIV_VALUES + WC_DIV_VALUES)
# [0s, 1s] for the locations
STATIC_SHIFT_VALUES = np.array(
    LANDSCAN_SHIFT_VALUES + [0, 0, 0] + DW_SHIFT_VALUES + WC_SHIFT_VALUES
)
STATIC_DIV_VALUES = np.array(LANDSCAN_DIV_VALUES + [1, 1, 1] + DW_DIV_VALUES + WC_DIV_VALUES)

SPACE_TIME_BANDS_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

TIME_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
        "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
        "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
        "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
        "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
    }
)


# this normalizing dict is sourced from
# https://github.com/nasaharvest/galileo/blob/main/config/normalization.json
# its used to normalize the data. The keys (e.g. "13") are used to query
# which tensor (e.g. space_time_x) is associated to the means and stds,
# where the key represents the number of dimensions in the tensor (i.e. x.shape[-1])
NORMALIZING_DICT = {
    "total_n": 127155,
    "sampled_n": 10000,
    "13": {
        "mean": [
            -11.728724389184965,
            -18.85558188024017,
            1395.3408730676722,
            1338.4026921784578,
            1343.09883810357,
            1543.8607982512297,
            2186.2022069512263,
            2525.0932853316694,
            2410.3377187373408,
            2750.2854646886753,
            2234.911100061487,
            1474.5311266077113,
            0.2892116502999044,
        ],
        "std": [
            4.887145774840316,
            5.730270320384293,
            917.7041440370853,
            913.2988423581528,
            1092.678723527555,
            1047.2206083460424,
            1048.0101611156767,
            1143.6903026819996,
            1098.979177731649,
            1204.472755085893,
            1145.9774063078878,
            980.2429840007796,
            0.2720939024500081,
        ],
    },
    "16": {
        "mean": [
            673.0152819503361,
            5.930092668915115,
            0.10470439140978786,
            0.23965913270066183,
            0.08158044385860364,
            0.04246976254259546,
            0.11304392863520317,
            0.17329647890362473,
            0.0698981691616277,
            0.12130267132802142,
            0.04671318615236216,
            10.973119802517362,
            1.0927069179958768,
            1.6991394232855903,
            0.03720594618055555,
            1.3671352688259548,
        ],
        "std": [
            983.0697298296237,
            8.167406789813247,
            0.18771647977504985,
            0.2368313455675914,
            0.08024268534756586,
            0.04045374496146404,
            0.11350342472061795,
            0.1279898111718168,
            0.12042341550438586,
            0.13602408145504347,
            0.043971116096060345,
            31.255340146970997,
            10.395974878206689,
            12.92380617159917,
            1.9285254295940466,
            11.612179775408928,
        ],
    },
    "6": {
        "mean": [
            271.5674963541667,
            0.08554303677156568,
            657.3181260091111,
            692.1291795806885,
            562.781331880633,
            1.5647115934036673,
        ],
        "std": [
            79.80828940314429,
            0.11669547098151486,
            704.0008695557707,
            925.0116126406431,
            453.2434022278578,
            7.513020170832818,
        ],
    },
    "18": {
        "mean": [
            188.20315880851746,
            0.2804946561574936,
            0.11371652073860168,
            0.058778801321983334,
            0.10474256777763366,
            0.2396918488264084,
            0.08152248692512512,
            0.04248040814399719,
            0.11303179881572724,
            0.17326324067115784,
            0.06998309404850006,
            0.12122812910079957,
            0.04671641788482666,
            10.98456594619751,
            1.0968475807189941,
            1.6947754135131836,
            0.03320046615600586,
            1.3602827312469483,
        ],
        "std": [
            1154.5919128300602,
            0.5276998078079327,
            0.7021637331734328,
            0.36528892213195063,
            0.17470213191865785,
            0.20411195416718833,
            0.0660782470089761,
            0.03380702424871257,
            0.09809195568521663,
            0.11292471052124119,
            0.09720748930233268,
            0.12912217763726777,
            0.0399973913151906,
            23.725471823867462,
            5.715238079725388,
            9.030481416228302,
            0.9950220242487364,
            7.754429123862099,
        ],
    },
}


class Normalizer:
    """Normalize Galileo inputs."""

    std_bands: dict[int, list] = {
        # we exclude NDVI because its already between 0 and 1, so we don't
        # want to apply further normalization to it.
        len(SPACE_TIME_BANDS): [b for b in SPACE_TIME_BANDS if b != "NDVI"],
        len(SPACE_BANDS): SRTM_BANDS,
        len(TIME_BANDS): TIME_BANDS,
        len(STATIC_BANDS): LANDSCAN_BANDS,
    }

    def __init__(self, std_multiplier: float = 2):
        """Normalize Galileo inputs.

        Args:
            std_multiplier: std_multiplier to apply
        """
        name_to_bands = {
            len(SPACE_TIME_BANDS): SPACE_TIME_BANDS,
            len(SPACE_BANDS): SPACE_BANDS,
            len(TIME_BANDS): TIME_BANDS,
            len(STATIC_BANDS): STATIC_BANDS,
        }
        self.shift_div_dict = {
            len(SPACE_TIME_BANDS): {
                "shift": deepcopy(SPACE_TIME_SHIFT_VALUES),
                "div": deepcopy(SPACE_TIME_DIV_VALUES),
            },
            len(SPACE_BANDS): {
                "shift": deepcopy(SPACE_SHIFT_VALUES),
                "div": deepcopy(SPACE_DIV_VALUES),
            },
            len(TIME_BANDS): {
                "shift": deepcopy(TIME_SHIFT_VALUES),
                "div": deepcopy(TIME_DIV_VALUES),
            },
            len(STATIC_BANDS): {
                "shift": deepcopy(STATIC_SHIFT_VALUES),
                "div": deepcopy(STATIC_DIV_VALUES),
            },
        }
        for key_as_str, val in NORMALIZING_DICT.items():
            if "n" in key_as_str:
                continue
            key = int(key_as_str)
            bands_to_replace = self.std_bands[key]
            for band in bands_to_replace:
                band_idx = name_to_bands[key].index(band)
                mean = cast(dict[str, list], val)["mean"][band_idx]
                std = cast(dict[str, list], val)["std"][band_idx]
                min_value = mean - (std_multiplier * std)
                max_value = mean + (std_multiplier * std)
                div = max_value - min_value
                if div == 0:
                    raise ValueError(f"{band} has div value of 0")
                self.shift_div_dict[key]["shift"][band_idx] = min_value
                self.shift_div_dict[key]["div"][band_idx] = div

    @staticmethod
    def _normalize(
        x: torch.Tensor, shift_values: torch.Tensor, div_values: torch.Tensor
    ) -> torch.Tensor:
        x = (x - shift_values) / div_values
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalizer."""
        div_values = self.shift_div_dict[x.shape[-1]]["div"]
        shift_values = self.shift_div_dict[x.shape[-1]]["shift"]
        return self._normalize(x, shift_values, div_values)


DEFAULT_NORMALIZER = Normalizer()


class MaskedOutput(NamedTuple):
    """A masked output (i.e. an input to Galileo).

    A mask can take 3 values:
    0: seen by the encoder (i.e. makes the key and value tokens in the decoder)
    1: not seen by the encoder, and ignored by the decoder
    2: not seen by the encoder, and processed by the decoder (the decoder's query values)
    """

    s_t_x: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS)]
    sp_x: torch.Tensor  # [B, H, W, len(SPACE_BANDS)]
    t_x: torch.Tensor  # [B, T, len(TIME_BANDS)]
    st_x: torch.Tensor  # [B, len(STATIC_BANDS)]
    s_t_m: torch.Tensor  # [B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX)]
    sp_m: torch.Tensor  # [B, H, W, len(SPACE_BAND_GROUPS_IDX)]
    t_m: torch.Tensor  # [B, T, len(TIME_BAND_GROUPS_IDX)]
    st_m: torch.Tensor  # [B, len(STATIC_BAND_GROUPS_IDX)]
    months: torch.Tensor  # [B, T]


def construct_galileo_input(
    s1: torch.Tensor | None = None,  # [H, W, T, D]
    s2: torch.Tensor | None = None,  # [H, W, T, D]
    era5: torch.Tensor | None = None,  # [T, D]
    tc: torch.Tensor | None = None,  # [T, D]
    viirs: torch.Tensor | None = None,  # [T, D]
    srtm: torch.Tensor | None = None,  # [H, W, D]
    dw: torch.Tensor | None = None,  # [H, W, D]
    wc: torch.Tensor | None = None,  # [H, W, D]
    landscan: torch.Tensor | None = None,  # [D]
    latlon: torch.Tensor | None = None,  # [D]
    months: torch.Tensor | None = None,  # [T]
    normalize: bool = False,
) -> MaskedOutput:
    """Construct a Galileo input."""
    space_time_inputs = [s1, s2]
    time_inputs = [era5, tc, viirs]
    space_inputs = [srtm, dw, wc]
    static_inputs = [landscan, latlon]
    devices = [
        x.device
        for x in space_time_inputs + time_inputs + space_inputs + static_inputs
        if x is not None
    ]

    if len(devices) == 0:
        raise ValueError("At least one input must be not None")
    if not all(devices[0] == device for device in devices):
        raise ValueError("Received tensors on multiple devices")
    device = devices[0]

    # first, check all the input shapes are consistent
    batch_list = (
        [x.shape[0] for x in space_time_inputs if x is not None]
        + [x.shape[0] for x in time_inputs if x is not None]
        + [x.shape[0] for x in space_inputs if x is not None]
        + [x.shape[0] for x in static_inputs if x is not None]
    )
    timesteps_list = [x.shape[3] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in time_inputs if x is not None
    ]
    height_list = [x.shape[1] for x in space_time_inputs if x is not None] + [
        x.shape[1] for x in space_inputs if x is not None
    ]
    width_list = [x.shape[2] for x in space_time_inputs if x is not None] + [
        x.shape[2] for x in space_inputs if x is not None
    ]
    b = 1
    if len(batch_list) > 0:
        if len(set(batch_list)) > 1:
            raise ValueError("Inconsistent number of batch sizes per input")
        b = batch_list[0]

    if len(timesteps_list) > 0:
        if not all(timesteps_list[0] == timestep for timestep in timesteps_list):
            raise ValueError("Inconsistent number of timesteps per input")
        t = timesteps_list[0]
    else:
        t = 1
    if len(height_list) > 0:
        if not all(height_list[0] == height for height in height_list):
            raise ValueError("Inconsistent heights per input")
        if not all(width_list[0] == width for width in width_list):
            raise ValueError("Inconsistent widths per input")
        h = height_list[0]
        w = width_list[0]
    else:
        h, w = 1, 1

    # now, we can construct our empty input tensors. By default, everything is masked
    s_t_x = torch.zeros((b, h, w, t, len(SPACE_TIME_BANDS)), dtype=torch.float, device=device)
    s_t_m = torch.ones(
        (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)),
        dtype=torch.float,
        device=device,
    )
    sp_x = torch.zeros((b, h, w, len(SPACE_BANDS)), dtype=torch.float, device=device)
    sp_m = torch.ones((b, h, w, len(SPACE_BAND_GROUPS_IDX)), dtype=torch.float, device=device)
    t_x = torch.zeros((b, t, len(TIME_BANDS)), dtype=torch.float, device=device)
    t_m = torch.ones((b, t, len(TIME_BAND_GROUPS_IDX)), dtype=torch.float, device=device)
    st_x = torch.zeros((b, len(STATIC_BANDS)), dtype=torch.float, device=device)
    st_m = torch.ones((b, len(STATIC_BAND_GROUPS_IDX)), dtype=torch.float, device=device)

    for x, bands_list, group_key in zip([s1, s2], [S1_BANDS, S2_BANDS], ["S1", "S2"]):
        if x is not None:
            indices = [idx for idx, val in enumerate(SPACE_TIME_BANDS) if val in bands_list]
            groups_idx = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if group_key in key
            ]
            s_t_x[:, :, :, :, indices] = x
            s_t_m[:, :, :, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [srtm, dw, wc], [SRTM_BANDS, DW_BANDS, WC_BANDS], ["SRTM", "DW", "WC"]
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(SPACE_BANDS) if val in bands_list]
            groups_idx = [idx for idx, key in enumerate(SPACE_BAND_GROUPS_IDX) if group_key in key]
            sp_x[:, :, :, indices] = x
            sp_m[:, :, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [era5, tc, viirs],
        [ERA5_BANDS, TC_BANDS, VIIRS_BANDS],
        ["ERA5", "TC", "VIIRS"],
    ):
        if x is not None:
            indices = [idx for idx, val in enumerate(TIME_BANDS) if val in bands_list]
            groups_idx = [idx for idx, key in enumerate(TIME_BAND_GROUPS_IDX) if group_key in key]
            t_x[:, :, indices] = x
            t_m[:, :, groups_idx] = 0

    for x, bands_list, group_key in zip(
        [landscan, latlon], [LANDSCAN_BANDS, LOCATION_BANDS], ["LS", "location"]
    ):
        if x is not None:
            if group_key == "location":
                # transform latlon to cartesian
                x = cast(torch.Tensor, to_cartesian(x[:, 0], x[:, 1]))
            indices = [idx for idx, val in enumerate(STATIC_BANDS) if val in bands_list]
            groups_idx = [idx for idx, key in enumerate(STATIC_BAND_GROUPS_IDX) if group_key in key]
            st_x[:, indices] = x
            st_m[:, groups_idx] = 0

    if months is None:
        months = torch.ones((b, t), dtype=torch.long, device=device) * DEFAULT_MONTH
    elif months.shape[1] != t:
        raise ValueError("Incorrect number of input months")

    if normalize:
        s_t_x = torch.from_numpy(DEFAULT_NORMALIZER(s_t_x.cpu().numpy())).to(device).float()
        sp_x = torch.from_numpy(DEFAULT_NORMALIZER(sp_x.cpu().numpy())).to(device).float()
        t_x = torch.from_numpy(DEFAULT_NORMALIZER(t_x.cpu().numpy())).to(device).float()
        st_x = torch.from_numpy(DEFAULT_NORMALIZER(st_x.cpu().numpy())).to(device).float()

    return MaskedOutput(
        s_t_x=s_t_x,
        s_t_m=s_t_m,
        sp_x=sp_x,
        sp_m=sp_m,
        t_x=t_x,
        t_m=t_m,
        st_x=st_x,
        st_m=st_m,
        months=months,
    )


def to_cartesian(
    lat: float | np.ndarray | torch.Tensor, lon: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Transform latitudes and longitudes to cartesian coordinates."""
    if isinstance(lat, float):
        assert -90 <= lat <= 90, f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        assert -180 <= lon <= 180, f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        assert isinstance(lon, float), f"Expected float got {type(lon)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return np.array([x, y, z])
    elif isinstance(lon, np.ndarray):
        assert -90 <= lat.min(), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert 90 >= lat.max(), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert -180 <= lon.min(), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert 180 >= lon.max(), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_np = np.cos(lat) * np.cos(lon)
        y_np = np.cos(lat) * np.sin(lon)
        z_np = np.sin(lat)
        return np.stack([x_np, y_np, z_np], axis=-1)
    elif isinstance(lon, torch.Tensor):
        assert -90 <= lat.min(), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert 90 >= lat.max(), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert -180 <= lon.min(), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert 180 >= lon.max(), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, torch.Tensor), f"Expected torch.Tensor got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_t = torch.cos(lat) * torch.cos(lon)
        y_t = torch.cos(lat) * torch.sin(lon)
        z_t = torch.sin(lat)
        return torch.stack([x_t, y_t, z_t], dim=-1)
    else:
        raise AssertionError(f"Unexpected input type {type(lon)}")
