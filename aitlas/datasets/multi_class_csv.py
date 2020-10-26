import csv
import os

import torchvision.transforms as transforms

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import MultiClassCsvDatasetSchema


class MultiClassCsvDataset(BaseDataset):
    schema = MultiClassCsvDatasetSchema

    classes_to_idx = None  # need to put your mapping here

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)

    def prepare(self):
        self.data = self.load_dataset(self.config.csv_file_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = image_loader(self.data[index][0])
        # apply transformations
        img = self.transform(img)
        target = self.data[index][1]
        return img, target

    def __len__(self):
        return len(self.data)

    def labels(self):
        return list(self.classes_to_idx.keys())

    def load_dataset(self, file_path):
        if not self.classes_to_idx:
            raise ValueError(
                "You need to implement the classes to index mapping for the dataset"
            )
        data = []
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for index, row in enumerate(csv_reader):
                path = row[0]
                item = (path, self.classes_to_idx[row[1]])
                data.append(item)
        return data
