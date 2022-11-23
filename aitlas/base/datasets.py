import torch
from torch.utils.data import Dataset

from .config import Configurable
from .schemas import BaseDatasetSchema
from .transforms import load_transforms


class BaseDataset(Dataset, Configurable):

    schema = BaseDatasetSchema
    labels = None  # need to put the labels here
    name = None

    def __init__(self, config):
        Dataset.__init__(self)
        Configurable.__init__(self, config)

        # get dataloader parameters
        self.shuffle = self.config.shuffle
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.pin_memory = self.config.pin_memory

        # get labels if provided in config and not set in class
        if not self.labels and self.config.labels:
            self.labels = self.config.labels

        # get the transformations to be applied for the image and for the target
        self.transform = self.load_transforms(self.config.transforms)
        self.target_transform = self.load_transforms(self.config.target_transforms)
        self.joint_transform = self.load_transforms(self.config.joint_transforms)

    def __getitem__(self, index):
        """ Implement here what you want to return"""
        raise NotImplementedError(
            "Please implement the `__getittem__` method for your dataset"
        )

    def __len__(self):
        raise NotImplementedError(
            "Please implement the `__len__` method for your dataset"
        )

    def get_name(self):
        if self.name:
            return self.name
        else:
            return ""

    def prepare(self):
        """Implement if something needs to happen to the dataset after object creation"""
        return True

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # drop_last=True,
        )

    def get_labels(self):
        """Implement this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `labels` method for your dataset"
        )

    def show_batch(self, size):
        """Implement this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `show_batch` method for your dataset"
        )

    def show_samples(self):
        """Implement this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `show_samples` method for your dataset"
        )

    def show_image(self, index):
        """Implement this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `show_image` method for your dataset"
        )

    def data_distribution_table(self):
        """Implement this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `data_distribution_table` method for your dataset"
        )

    def data_distribution_barchart(self):
        """Implement this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `data_distribution_barchart` method for your dataset"
        )

    def load_transforms(self, class_names):
        """Loads transformation classes and make a composition of them"""
        return load_transforms(class_names, self.config)

