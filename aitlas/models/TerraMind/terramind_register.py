from functools import partial
from torch import nn
from .tm_utils import LayerNorm
from .terramind import TerraMindModule
from .terramind_vit import TerraMindViT
from .terramind_tim import TerraMindTiM


PRETRAINED_BANDS = {
    "untok_sen2l2a@224": [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
    ],
    "untok_sen2l1c@224": [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "CIRRUS",
        "SWIR_1",
        "SWIR_2",
    ],
    "untok_sen2rgb@224": ["RED", "GREEN", "BLUE"],
    "untok_sen1grd@224": ["VV", "VH"],
    "untok_sen1rtc@224": ["VV", "VH"],
    "untok_dem@224": ["DEM"],
}

v1_pretraining_mean = {
    "untok_sen2l2a@224": [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648],
    "untok_sen2l1c@224": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.56, 3306.481, 1473.847,  506.07, 2472.825, 1838.929],
    "untok_sen2rgb@224": [87.271, 80.931, 66.667],
    "untok_sen1grd@224": [-12.599, -20.293],
    "untok_sen1rtc@224": [-10.93, -17.329],
    "untok_dem@224": [670.665],
    "tok_sen1grd@224": [-12.599, -20.293],
    "tok_sen1rtc@224": [-10.93, -17.329],
    "tok_sen2l2a@224": [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648],
    "tok_lulc@224": [0],
    "tok_dem@224": [670.665],
    "tok_ndvi@224": [0.327],
}

v1_pretraining_std = {
    "untok_sen2l2a@224": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    "untok_sen2l1c@224": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419],
    "untok_sen2rgb@224": [58.767, 47.663, 42.631],
    "untok_sen1grd@224": [5.195, 5.890],
    "untok_sen1rtc@224": [4.391, 4.459],
    "untok_dem@224": [951.272],
    "tok_sen2l2a@224": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
    "tok_sen1grd@224": [5.195, 5.890],
    "tok_sen1rtc@224": [4.391, 4.459],
    "tok_lulc@224": [1],
    "tok_dem@224": [951.272],
    "tok_ndvi@224": [0.322],
}

'''tokenizer_dict = {
    "v1": {
        "tok_sen2l2a@224": terramind_v1_tokenizer_s2l2a,
        "tok_sen1rtc@224": terramind_v1_tokenizer_s1rtc,
        "tok_sen1grd@224": terramind_v1_tokenizer_s1grd,
        "tok_dem@224": terramind_v1_tokenizer_dem,
        "tok_lulc@224": terramind_v1_tokenizer_lulc,
        "tok_ndvi@224": terramind_v1_tokenizer_ndvi,
        "coords": terramind_v1_coords_tokenizer,
    }
}'''


def terramind_v1_tiny(**kwargs):
    model = TerraMindViT(
        encoder_depth=12,
        dim=192,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        num_register_tokens=0,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.GELU,
        gated_mlp=False,
        #pretrained_bands=PRETRAINED_BANDS, TODO: Implement this and checkpoint_filter_fn_generate
        #tokenizer_dict=tokenizer_dict['v1'],
        #**kwargs
    )
    return model

def terramind_v1_small(**kwargs):
    model = TerraMindViT(
        encoder_depth=12,
        dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        proj_bias=True,
        mlp_bias=True,
        num_register_tokens=0,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.GELU,
        gated_mlp=False,
        #pretrained_bands=PRETRAINED_BANDS, TODO
        #tokenizer_dict=tokenizer_dict['v1'],
        #**kwargs
    )
    return model

def terramind_v1_base(**kwargs):
    model = TerraMindViT(
        encoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        num_register_tokens=0,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        #pretrained_bands=PRETRAINED_BANDS, TODO
        #tokenizer_dict=tokenizer_dict['v1'],
        #**kwargs
    )
    return model

def terramind_v1_large(**kwargs):
    model = TerraMindViT(
        encoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        num_register_tokens=0,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        #pretrained_bands=PRETRAINED_BANDS, TODO
        #tokenizer_dict=tokenizer_dict["v1"],
        #**kwargs
    )
    return model