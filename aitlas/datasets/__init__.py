# Datasets for semantic segmentation
from .aid import AIDDataset
from .aid_multilabel import AIDMultiLabelDataset
from .airs import AIRSDataset
from .amazon_rainforest import AmazonRainforestDataset
from .big_earth_net import BigEarthNetDataset
from .brazilian_coffee_scenes import BrazilianCoffeeScenesDataset

# Datasets for crop type prediction
from .breizhcrops import BreizhCropsDataset
from .camvid import CamVidDataset
from .chactun import ChactunDataset
from .clrs import CLRSDataset
from .dfc15_multilabel import DFC15MultiLabelDataset
from .eopatch_crops import EOPatchCrops
from .eurosat import EurosatDataset
from .inria import InriaDataset
from .landcover_ai import LandCoverAiDataset
from .massachusetts_buildings import MassachusettsBuildingsDataset
from .massachusetts_roads import MassachusettsRoadsDataset
from .mlrs_net import MLRSNetMultiLabelDataset
from .multiclass_classification import MultiClassClassificationDataset

# Multi class and multi label datasets for land use and land cover classification
from .multilabel_classification import MultiLabelClassificationDataset

# Datasets for object detection
from .object_detection import ObjectDetectionCocoDataset, ObjectDetectionPascalDataset
from .optimal_31 import Optimal31Dataset
from .pattern_net import PatternNetDataset
from .planet_uas import PlanetUASMultiLabelDataset
from .resisc45 import Resisc45Dataset
from .rsd46_whu import RSD46WHUDataset
from .rsi_cb256 import RSICB256Dataset
from .rsscn7 import RSSCN7Dataset
from .sat6 import SAT6Dataset
from .semantic_segmentation import SemanticSegmentationDataset
from .siri_whu import SiriWhuDataset
from .so2sat import So2SatDataset
from .uc_merced import UcMercedDataset
from .uc_merced_multilabel import UcMercedMultiLabelDataset
from .whu_rs19 import WHURS19Dataset
from .npz import NpzDataset
