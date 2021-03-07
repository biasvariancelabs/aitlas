import csv
import os
import numpy as np

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema

CLASSES_TO_IDX = {'sky': 0, 'building': 1, 'column_pole': 2, 'road': 3, 'sidewalk': 4, 'tree': 5, 'sign': 6, 'fence': 7,
                  'car': 8, 'pedestrian': 9, 'byciclist': 10, 'void': 11}


class CamVidDataset(BaseDataset):

    schema = SegmentationDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)
        self.images = []
        self.masks = []
        self.class_values = []

        self.load_dataset(self.config.root)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], False)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')
        image, mask = self.transform({"image": image, "mask": mask})
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, root_dir):
        self.class_values = CLASSES_TO_IDX.values()

        if not self.class_values:
            raise ValueError(
                "You need to implement the classes to index mapping for the dataset"
            )

        ids = os.listdir(os.path.join(root_dir, 'images'))
        self.images = [os.path.join(root_dir, 'images', image_id) for image_id in ids]
        self.masks = [os.path.join(root_dir, 'masks', image_id) for image_id in ids]

    def labels(self):
        return list(CLASSES_TO_IDX.keys())
