import numpy as np

from ..base import BaseDataset
from .schemas import NPZDatasetSchema
from numpy import load
from PIL import Image

"""
Load a dataset from a file in .npz format
The file contains the train, validation and the test splits 
"""


class NpzDataset(BaseDataset):
    schema = NPZDatasetSchema
    labels = None

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the data
        self.npz_file = self.config.npz_file
        self.labels = self.config.labels
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image and convert to RGB
        img, target = self.data[index]
        img = np.asarray(Image.fromarray(img).convert('RGB'))
        # apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        pass

    def data_distribution_barchart(self):
        pass

    def show_samples(self):
        pass

    def show_image(self, index):
        pass

    def show_batch(self, size, show_title=True):
        pass

    def load_dataset(self):
        data = []
        if self.npz_file:
            raw_data = load(self.npz_file)
            images = raw_data[f'{self.config.mode}_images']
            labels = raw_data[f'{self.config.mode}_labels']

            for index, image in enumerate(images):
                item = (
                    image,
                    labels[index][0],
                )
                data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data


