"""Dataset base class.

This is the base class for all datasets. All datasets should subclass it. 

"""

import torch
from torch.utils.data import Dataset

from .config import Configurable
from .schemas import BaseDatasetSchema
from .transforms import load_transforms


class BaseDataset(Dataset, Configurable):
    """This class represents a basic dataset for machine learning tasks. It is a 
    subclass of both :class:Dataset and :class:Configurable. 
    You can use it as a base class to define your own custom datasets.

    :param Dataset: _description_
    :type Dataset: _type_
    :param Configurable: _description_
    :type Configurable: _type_
    """

    schema = BaseDatasetSchema
    labels = None  # need to put the labels here
    name = None

    def __init__(self, config):
        """ Initialize the dataset with the given configuration.

        :param config: _description_
        :type config: _type_
        """
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
        """ Abstract method to return an item from the dataset at the given index.

                    :param index: 
        """
        raise NotImplementedError(
            "Please implement the `__getittem__` method for your dataset"
        )

    def __len__(self):
        """Abstract method to return the length of the dataset."""
        raise NotImplementedError(
            "Please implement the `__len__` method for your dataset"
        )

    def get_name(self):
        """Return the name of the dataset, if set."""
        if self.name:
            return self.name
        else:
            return ""

    def prepare(self):
        """Abstract method to prepare the dataset after object creation."""
        return True

    def dataloader(self):
        """Create a DataLoader for the dataset with the configured parameters.

        :return: Pytorch DataLoader
        :rtype: torch.utils.data.DataLoader
        """
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # drop_last=True,
        )

    def get_labels(self):
        """Abstract method to return the complete set of labels of the dataset."""
        raise NotImplementedError(
            "Please implement the `labels` method for your dataset"
        )

    def show_batch(self, size):
        """Abstract method to display a batch of items from the dataset.

            :param size: The size of the batch to display."""
        
        raise NotImplementedError(
            "Please implement the `show_batch` method for your dataset"
        )

    def show_samples(self):
        """Abstract method to display a set of samples from the dataset."""
        raise NotImplementedError(
            "Please implement the `show_samples` method for your dataset"
        )

    def show_image(self, index):
        """Abstract method to display an image from the dataset at the given index.
        
        :param index: The index of the image to display."""
        raise NotImplementedError(
            "Please implement the `show_image` method for your dataset"
        )

    def data_distribution_table(self):
        """Abstract method to display a table with the data distribution of the dataset."""
        raise NotImplementedError(
            "Please implement the `data_distribution_table` method for your dataset"
        )

    def data_distribution_barchart(self):
        """Abstract method to display a data distribution of the dataset as a bar chart."""
        raise NotImplementedError(
            "Please implement the `data_distribution_barchart` method for your dataset"
        )

    def load_transforms(self, class_names):
        """Loads transformation classes and make a composition of them"""
        return load_transforms(class_names, self.config)

