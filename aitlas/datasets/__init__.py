# Datasets for semantic segmentation
from .aeroscapes import AeroscapesDataset
from .aid import AIDDataset
from .aid_multilabel import AIDMultiLabelDataset
from .airs import AIRSDataset
from .amazon_rainforest import AmazonRainforestDataset
from .big_earth_net import BigEarthNetDataset
from .brazilian_coffee_scenes import BrazilianCoffeeScenesDataset

# Datasets for crop type prediction
from .breizhcrops import BreizhCropsDataset
from .camvid import CamVidDataset
from .cesbio import CESBIODataset
from .cesbio_ai4qc import CESBIO_AI4QCDataset
from .chactun import ChactunDataset
from .cloudsen12 import CloudSEN12Dataset
from .clrs import CLRSDataset
from .deepglobe import DeepGlobeDataset
from .dfc15_multilabel import DFC15MultiLabelDataset
from .eopatch_crops import EOPatchCrops
from .eurosat import EurosatDataset
from .floodnet import FloodNetDataset
from .gid15 import GID15Dataset
from .gsfc import GSFCDataset
from .gsfc_ai4qc import GSFC_AI4QCDataset
from .gta_v_sid import GTADataset
from .hollstein import HollsteinDataset
from .hollstein_ai4qc import Hollstein_AI4QCDataset
from .inria import InriaDataset
from .kappaset import KappaSetDataset
from .kappaset_ai4qc import KappaSet_AI4QCDataset
from .kappazeta import KappaZetaDataset
from .landcover_ai import LandCoverAiDataset
from .loveda import LoveDADataset
from .massachusetts_buildings import MassachusettsBuildingsDataset
from .massachusetts_roads import MassachusettsRoadsDataset
from .mlrs_net import MLRSNetMultiLabelDataset
from .multiclass_classification import MultiClassClassificationDataset

# Multi class and multi label datasets for land use and land cover classification
from .multilabel_classification import MultiLabelClassificationDataset
from .npz import NpzDataset

# Datasets for object detection
from .object_detection import ObjectDetectionCocoDataset, ObjectDetectionPascalDataset
from .optimal_31 import Optimal31Dataset
from .parallax_ai4qc import (
    ObjectDetectionRotatedParallaxDataset,
    ObjectDetectionStraightParallaxDataset,
)
from .pattern_net import PatternNetDataset
from .pixbox_ai4qc import PixBox_AI4QCDataset
from .planet_uas import PlanetUASMultiLabelDataset
from .potsdam import PotsdamDataset
from .resisc45 import Resisc45Dataset
from .rfi_ai4qc import (
    ObjectDetectionRfiCocoDataset,
    ObjectDetectionRfiPascalDataset,
    ObjectDetectionRfiYoloDataset,
)
from .rsd46_whu import RSD46WHUDataset
from .rsi_cb256 import RSICB256Dataset
from .rsscn7 import RSSCN7Dataset
from .s1anomalies_ai4qc import S1NewAnomaliesDataset
from .s2anomalies_ai4qc import S2NewAnomaliesDataset
from .sat6 import SAT6Dataset
from .semantic_segmentation import SemanticSegmentationDataset
from .siri_whu import SiriWhuDataset
from .so2sat import So2SatDataset
from .spacenet6 import SpaceNet6Dataset
from .uavid import UAVidDataset
from .uc_merced import UcMercedDataset
from .uc_merced_multilabel import UcMercedMultiLabelDataset
from .whu_aerial import WHUAerialDataset
from .whu_mix import WHUMixDataset
from .whu_rs19 import WHURS19Dataset
