import os

import numpy as np

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema


LABELS = [
    "sky",
    "building",
    "column_pole",
    "road",
    "sidewalk",
    "tree",
    "sign",
    "fence",
    "car",
    "pedestrian",
    "byciclist",
    "void",
]

"""
For the CamVid dataset the mask is in one file, each label is color coded.
"""


class CamVidDataset(BaseDataset):
    schema = SegmentationDatasetSchema
    labels = LABELS

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.data_dir)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], False)
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, data_dir):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "masks", image_id) for image_id in ids]

    def get_labels(self):
        return self.labels
