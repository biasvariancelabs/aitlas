import csv
import os
import numpy as np

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema

#"Background": 0
#"Buildings": 1
#"Woodlands": 2
#"Water": 3

CLASSES_TO_IDX = {"Buildings": 1}

class SegmentationDataset(BaseDataset):

    schema = SegmentationDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)
        self.images = []
        self.masks = []
        self.class_values = []

        self.load_dataset(self.config.root, self.config.csv_file_path)


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], True)

        # extract certain classes from mask (e.g. Buildings)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')
        image, mask = self.transform({"image": image, "mask": mask})
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, root_dir, file_path):
        self.class_values = CLASSES_TO_IDX.values()

        if not self.class_values:
            raise ValueError(
                "You need to implement the classes to index mapping for the dataset"
            )

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for index, row in enumerate(csv_reader):
                self.images.append(os.path.join(root_dir, row[0] + '.jpg'))
                self.masks.append(os.path.join(root_dir, row[0] + '_m.png'))

    def labels(self):
        return list(CLASSES_TO_IDX.keys())


