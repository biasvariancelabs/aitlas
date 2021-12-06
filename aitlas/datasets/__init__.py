# Multi class datasets for land use and land cover classification
from .multiclass_classification import MultiClassClassificationDataset
from .aid import AIDDataset
from .clrs import CLRSDataset
from .eurosat import EurosatDataset
from .pattern_net import PatternNetDataset
from .resisc45 import Resisc45Dataset
from .rsd46_whu import RSD46WHUDataset
from .rsi_cb256 import RSICB256Dataset
from .rsscn7 import RSSCN7Dataset
from .sat6 import SAT6Dataset
from .siri_whu import SiriWhuDataset
from .uc_merced import UcMercedDataset
from .whu_rs19 import WHURS19Dataset

# Multi label datasets for land use and land cover classification
from .multilabel_classification import MultiLabelClassificationDataset
from .aid_multilabel import AIDMultiLabelDataset
from .big_earth_net import BigEarthNetDataset
from .dfc15_multilabel import DFC15MultiLabelDataset
from .uc_merced_multilabel import UcMercedMultiLabelDataset
from .mlrs_net import MLRSNetMultiLabelDataset
from .planet_uas import PlanetUASMultiLabelDataset

# Datasets for semantic segmentation
from .amazon_rainforest import AmazonRainforestDataset
from .airs import AIRSDataset
from .camvid import CamVidDataset
from .spacenet6 import SpaceNet6Dataset
from .massachusetts_buildings import MassachusettsBuildingsDataset
from .massachusetts_roads import MassachusettsRoadsDataset
from .inria import InriaDataset
from .landcover_ai import LandCoverAiDataset
from .chactun import ChactunDataset

# Datasets for crop type prediction
from .crops_classification import CropsDataset
from .eopatch_crops import EOPatchCrops
from .breizhcrops import BreizhCropsDataset

