from .multiclass_classification import MultiClassClassificationDataset
from .multilabel_classification import MultiLabelClassificationDataset

from .big_earth_net import (
    BaseBigEarthNetDataset,
    BigEarthNetAllBandsDataset,
    BigEarthNetRGBCsvDataset,
    BigEarthNetRGBDataset,
)
from .uc_merced_multilabel import UcMercedMultiLabelDataset
from .aid_multilabel import AIDMultiLabelDataset
from .dfc15_multilabel import DFC15MultiLabelDataset
from .mlrs_net import MLRSNet

from .eurosat import EurosatDataset
from .pattern_net import PatternNetDataset
from .resisc45 import Resisc45Dataset
from .uc_merced import UcMercedDataset
from .aid import AIDDataset
from .rsscn7 import RSSCN7Dataset
from .rsi_cb256 import RSICB256Dataset
from .siri_whu import SiriWhuDataset
from .sat6 import SAT6
from .clrs import CLRSDataset
from .whu_rs19 import WHURS19Dataset
from .rsd46_whu import RSD46WHUDataset

from .camvid import CamVidDataset
from .chactun import ChactunDataset
from .landcover_ai import LandCoverAiDataset
