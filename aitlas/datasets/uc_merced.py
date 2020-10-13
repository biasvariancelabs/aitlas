import os

import torchvision.transforms as transforms

from ..base import DatasetFolderMixin, SplitableDataset
from ..utils import pil_loader, tiff_loader
from .schemas import UcMercedDatasetSchema


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


class UcMercedDataset(SplitableDataset, DatasetFolderMixin):
    schema = UcMercedDatasetSchema

    url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"

    classes_to_idx = CLASSES_TO_IDX

    def __init__(self, config):
        # now call the constuctor to validate the schema and split the data
        SplitableDataset.__init__(self, config)
        self.image_loader = tiff_loader
        self.data = self.make_dataset(self.config.root)

    def load_transforms(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = self.image_loader(self.data[index][0])
        # apply transformations
        img = self.transform(img)
        target = self.data[index][1]
        return img, target

    def __len__(self):
        return len(self.data)

    def labels(self):
        return list(self.classes_to_idx.keys())
