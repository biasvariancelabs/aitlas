import os

import torchvision.transforms as transforms

from ..utils import pil_loader, tiff_loader
from .multi_class_csv import MultiClassCsvDataset


CLASSES_TO_IDX = {
    "agricultural": 0,
    "airplane": 1,
    "baseballdiamond": 2,
    "beach": 3,
    "buildings": 4,
    "chaparral": 5,
    "denseresidential": 6,
    "forest": 7,
    "freeway": 8,
    "golfcourse": 9,
    "harbor": 10,
    "intersection": 11,
    "mediumresidential": 12,
    "mobilehomepark": 13,
    "overpass": 14,
    "parkinglot": 15,
    "river": 16,
    "runway": 17,
    "sparseresidential": 18,
    "storagetanks": 19,
    "tenniscourt": 20,
}


class UcMercedDataset(MultiClassCsvDataset):

    classes_to_idx = CLASSES_TO_IDX
