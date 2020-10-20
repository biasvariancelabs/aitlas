import os
import os.path
import csv
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms

from .config import Configurable
from .schemas import BaseDatasetSchema, SplitableDatasetSchema, CsvDatasetSchema


class BaseDataset(Dataset, Configurable):

    schema = BaseDatasetSchema

    train_indices = []
    test_indices = []
    val_indices = []

    train_indices_inverted = []

    def __init__(self, config):
        Dataset.__init__(self)
        Configurable.__init__(self, config)

        # get the transformations to be applied
        self.transform = self.load_transforms()

    def __getitem__(self, index):
        """ Implement here what you want to return"""
        raise NotImplementedError(
            "Please implement the `__getittem__` method for your dataset"
        )

    def __len__(self):
        raise NotImplementedError(
            "Please implement the `__len__` method for your dataset"
        )

    def get_item_name(self, index):
        """Implement this method if you want to export splits"""
        raise NotImplementedError(
            "Please implement the `get_item_path` method for your dataset"
        )

    def prepare(self):
        """Implement if something needs to happen to the dataset after object creation"""
        return True

    def load_transforms(self):
        """Transformations that might be applied on the dataset"""
        return transforms.Compose([])

    def dataloader(self, dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )

    def train_loader(self):
        return self.dataloader(self)

    def val_loader(self):
        return None  # by default we think people won't want to to use a validation set

    def test_loader(self):
        return self.dataloader(self)

    def labels(self):
        """Implent this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `labels` method for your dataset"
        )


class SplitableDataset(BaseDataset):
    """General class for a dataset that can be split into train, test and validation"""

    schema = SplitableDatasetSchema

    def __init__(self, config):
        BaseDataset.__init__(self, config)

    def prepare(self):
        self.split()

    def split(self):
        should_split = True
        if not self.config.override:  # check if the files exists
            self.verify_files()

            # load splits
            self.train_indices = self.read_file_indices(self.config.split.train.file)
            self.test_indices = self.read_file_indices(self.config.split.test.file)
            if self.has_val():
                self.val_indices = self.read_file_indices(self.config.split.val.file)

            should_split = False

        if should_split:
            # check if the split is valid
            if not self.is_split_valid():
                raise ValueError(
                    "The defined split is invalid. The sum should be equal to 100."
                )
            # split the dataset
            self.make_splits()
            # save the splits
            self.save_splits()

        # create subsets from splits
        self.train_set = Subset(dataset=self, indices=self.train_indices)
        self.test_set = Subset(dataset=self, indices=self.test_indices)
        self.val_set = Subset(dataset=self, indices=self.val_indices)

    def has_val(self):
        return self.config.split.val

    def check_file(self, file):
        if not os.path.isfile(file):
            raise ValueError(f"Specified split file does not exist: {file}")
        return True

    def verify_files(self):
        is_valid = self.check_file(self.config.split.train.file)
        is_valid = is_valid and self.check_file(self.config.split.test.file)
        if self.has_val():
            is_valid = is_valid and self.check_file(self.config.split.val.file)
        return is_valid

    def read_file_indices(self, file):
        with open(file, "r") as f:
            return list(map(lambda x: int(x), f.read().splitlines()))

    def is_split_valid(self):
        res = self.config.split.train.ratio + self.config.split.test.ratio
        if self.has_val():
            res += self.config.split.val.ratio
        return res == 100

    def make_splits(self):
        size = self.__len__()
        train_num = int(size * self.config.split.train.ratio / 100)
        test_num = int(size * self.config.split.test.ratio / 100)

        arr_num = [train_num, test_num]

        if self.has_val():
            val_num = int(size * self.config.split.val.ratio / 100)
            arr_num.append(val_num)

        # fix roundup cases
        arr_num[0] += size - sum(arr_num)

        result = random_split(range(size), arr_num)

        self.train_indices = result[0]
        self.test_indices = result[1]

        if self.has_val():
            self.val_indices = result[2]

    def save_split(self, indices, file):
        with open(file, "w") as f:
            for ind in indices:
                f.write(f"{ind}\n")

    def save_splits(self):
        self.save_split(self.train_indices, self.config.split.train.file)
        self.save_split(self.test_indices, self.config.split.test.file)
        if self.has_val():
            self.save_split(self.val_indices, self.config.split.val.file)

    def train_loader(self):
        return self.dataloader(self.train_set, self.config.shuffle)

    def val_loader(self):
        return self.dataloader(self.val_set, False)

    def test_loader(self):
        return self.dataloader(self.test_set, False)


class CsvDataset(BaseDataset):
    schema = CsvDatasetSchema

    def __init__(self, config):
        BaseDataset.__init__(self, config)

    def prepare(self):
        self.read_csv()

    def read_csv(self):
        if self.config.train_csv:
            with open(self.config.train_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for index, row in enumerate(csv_reader):
                    self.train_indices.append(index)

        if self.config.val_csv:
            with open(self.config.val_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for index, row in enumerate(csv_reader):
                    self.val_indices.append(len(self.train_indices) + index)

        if self.config.test_csv:
            with open(self.config.test_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for index, row in enumerate(csv_reader):
                    self.test_indices.append(len(self.train_indices) + len(self.val_indices) + index)

        print(len(self.train_indices), len(self.val_indices), len(self.test_indices))

        # create subsets from csv files
        self.train_set = Subset(dataset=self, indices=self.train_indices)
        self.test_set = Subset(dataset=self, indices=self.test_indices)
        self.val_set = Subset(dataset=self, indices=self.val_indices)

    def train_loader(self):
        return self.dataloader(self.train_set, self.config.shuffle)

    def val_loader(self):
        return self.dataloader(self.val_set, False)

    def test_loader(self):
        return self.dataloader(self.test_set, False)

class DatasetFolderMixin:
    """A mixin for datasets the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    """

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        "webp",
    ]

    classes_to_idx = None  # need to put your mapping here

    def has_file_allowed_extension(self, filename, extensions):
        """Checks if a file is an allowed extension.
        Args:
            filename (string): path to a file
            extensions (iterable of strings): extensions to consider (lowercase)
        Returns:
            bool: True if the filename ends with one of given extensions
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)

    def is_image_file(self, filename):
        """Checks if a file is an allowed image extension.
        Args:
            filename (string): path to a file
        Returns:
            bool: True if the filename ends with a known image extension
        """
        return self.has_file_allowed_extension(filename, self.extensions)

    def make_dataset(self, dir, extensions=None):
        if not self.classes_to_idx:
            raise ValueError(
                "You need to implement the classes to index mapping for the dataset"
            )
        if not extensions:
            extensions = self.extensions
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(self.classes_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            # this ensures the image always have the same index numbers
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, self.classes_to_idx[target])
                        images.append(item)

        return images
