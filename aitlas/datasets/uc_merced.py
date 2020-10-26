import os

import torchvision.transforms as transforms

from .multi_class_csv import MultiClassCsvDataset
from ..utils import pil_loader, tiff_loader


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

    def __init__(self, config):
        # now call the constuctor to validate the schema and split the data
        MultiClassCsvDataset.__init__(self, config)

    def load_transforms(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
