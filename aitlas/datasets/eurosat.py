import os

import torchvision.transforms as transforms

from ..base import BaseTransforms, DatasetFolderMixin, SplitableDataset
from ..utils import pil_loader, tiff_loader
from .schemas import EurosatDatasetSchema


CLASSES_TO_IDX = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}


class EurosatDataset(SplitableDataset, DatasetFolderMixin):
    schema = EurosatDatasetSchema

    url_rgb = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
    url_allband = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"

    classes_to_idx = CLASSES_TO_IDX

    def __init__(self, config):
        # now call the constuctor to validate the schema and split the data
        SplitableDataset.__init__(self, config)

        if self.config.mode == "rgb":
            self.image_loader = pil_loader
            extensions = [".jpg", ".png"]
        elif self.config.mode == "all":
            self.image_loader = tiff_loader
            extensions = [".tif", ".tiff"]
        else:
            raise ValueError(
                "Eurosat mode invalid. It should be either `rgb` or `all`."
            )

        self.data = self.make_dataset(self.config.root, extensions)

        self.transform = self.train_transform()

    def default_transform(self):
        return EurosatTransforms({})

    def get_item_name(self, index):
        return self.data[index][0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = self.image_loader(self.data[index][0])
        target = self.data[index][1]
        return img, target

    def __len__(self):
        return len(self.data)

    def labels(self):
        return list(self.classes_to_idx.keys())


class EurosatTransforms(BaseTransforms):
    def __init__(self, config):
        BaseTransforms.__init__(self, config)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, input, target):
        return self.transform(input), target
