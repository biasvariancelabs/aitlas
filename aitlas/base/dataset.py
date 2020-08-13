import os

import torch
from torch.utils.data import Dataset, Subset, random_split

from .config import Configurable
from .schemas import BaseDatasetSchema, SplitableDatasetSchema


class BaseDataset(Dataset, Configurable):
    schema = BaseDatasetSchema

    def __init__(self, config):
        Dataset.__init__(self)
        Configurable.__init__(self, config)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
        )

    def trainloader(self):
        return self.dataloader(self)

    def valloader(self):
        return self.dataloader(self)

    def testloader(self):
        return self.dataloader(self)


class SplitableDataset(BaseDataset):
    schema = SplitableDatasetSchema
    default_dir = "./data/"

    train_incides = []
    test_indices = []
    val_indices = []

    def __init__(self, config):
        BaseDataset.__init__(self, config)

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
            self.split()
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

    def split(self):
        size = self.__len__()
        train_num = int(size * self.config.split.train.ratio / 100)
        test_num = int(size * self.config.split.test.ratio / 100)

        if self.has_val():
            val_num = int(size * self.config.split.val.ratio / 100)

            [self.train_indices, self.val_indices, self.test_indices] = random_split(
                self, [train_num, val_num, test_num]
            )
        else:
            [self.train_indices, self.test_indices] = random_split(
                self, [train_num, test_num]
            )

    def save_split(self, indices, file):
        with open(file, "w") as f:
            for ind in indices:
                f.write(f"{ind}\n")

    def save_splits(self):
        self.save_split(self.train_indices, self.config.split.train.file)
        self.save_split(self.test_indices, self.config.split.test.file)
        if self.has_val():
            self.save_split(self.val_indices, self.config.split.val.file)

    def trainloader(self):
        return self.dataloader(self.train_set)

    def valloader(self):
        return self.dataloader(self.val_set)

    def testloader(self):
        return self.dataloader(self.test_set)
