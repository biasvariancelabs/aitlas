import logging
import numpy as np
import math

from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import random_split
from ..base import BaseModel, BaseTask
from ..utils import (
    load_aitlas_format_dataset,
    load_folder_per_class_dataset,
    load_voc_format_dataset,
)
from .schemas import SplitTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseSplitTask(BaseTask):
    """Base task meant to split dataset"""

    schema = SplitTaskSchema

    is_multilabel = False  # specify it's a multilabel dataset or not

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    ]

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file

    def run(self):
        logging.info("Loading data...")
        self.images = self.load_images(self.data_dir, self.csv_file)

        logging.info("Making splits...")

        # load the images and labels
        self.X = np.array([x[0] for x in self.images])
        self.y = np.array([y[1] for y in self.images])

        self.split()
        logging.info("And that's it!")

    def has_val(self):
        return self.config.split.val and self.config.split.val.ratio > 0

    def is_split_valid(self):
        res = self.config.split.train.ratio + self.config.split.test.ratio
        if self.has_val():
            res += self.config.split.val.ratio
        return res == 100

    def split(self):
        if not self.is_split_valid():
            raise ValueError(
                "The defined split is invalid. The sum should be equal to 100."
            )
        # split the dataset
        self.make_splits()

    def save_split(self, X, y, file):
        with open(file, "w") as f:
            if self.is_multilabel:
                row = "\t".join(self.header)
                f.write(f"{row}\n")

            for xx, yy in zip(X, y):
                if self.is_multilabel:
                    # save in VOC format again
                    img = xx[0] if isinstance(xx, np.ndarray) else xx
                    img = img[img.rfind("images") + 7 : img.rfind(".")]
                    row = "\t".join([str(int(i)) for i in yy])
                    f.write(f"{img}\t{row}\n")
                else:
                    f.write(f"{xx},{yy}\n")
            f.close()

    def load_images(self, data_dir, csv_file, extensions=None):
        """Attempts to read in VOC format, then in internal format, then in folder per class format"""
        images = []
        try:
            images = load_voc_format_dataset(data_dir, csv_file)

            # if this format is load, it's a multilabel dataset
            self.is_multilabel = True

            # read the header again. TODO: Maybe this can be a bit better implemented.
            with open(csv_file, "rb") as f:
                self.header = f.readline().decode("utf-8").strip().split("\t")

        except TypeError:  # it's not in VOC format, then let's try aitlas (CSV) internal one
            if csv_file is not None:
                images = load_aitlas_format_dataset(csv_file)
            else:
                if not extensions:
                    extensions = self.extensions
                images = load_folder_per_class_dataset(data_dir, extensions)

        if not images:
            raise ValueError("No images were found!")

        return images

    def make_splits(self):
        # load paths and labels
        test_size = float(self.config.split.test.ratio / 100)

        X_train, y_train, X_test, y_test = self.perform_split(self.X, self.y, test_size)

        # if there is a validation split, perform that as well
        if self.has_val():
            val_size = float(
                self.config.split.val.ratio
                / (self.config.split.val.ratio + self.config.split.train.ratio)
            )

            X_train, y_train, X_val, y_val = self.perform_split(
                X_train, y_train, val_size
            )

            # save split
            self.save_split(X_val, y_val, self.config.split.val.file)

        # save the other splits
        self.save_split(X_train, y_train, self.config.split.train.file)
        self.save_split(X_test, y_test, self.config.split.test.file)

    def perform_split(self, X, y, test_size):
        raise NotImplementedError


class RandomSplitTask(BaseSplitTask):
    """Randomly split a folder containing images"""

    def perform_split(self, X, y, test_size):
        """Peform actual split using pytorch random split"""
        size = len(X)
        train_num = int(math.ceil(size * (1 - test_size)))
        test_num = int(size * test_size)

        arr_num = [train_num, test_num]

        train_split, test_split = random_split(range(size), arr_num)

        X_train, y_train, X_test, y_test = [], [], [], []

        for i in train_split:
            X_train.append(X[i])
            y_train.append(y[i])

        for i in test_split:
            X_test.append(X[i])
            y_test.append(y[i])

        return X_train, y_train, X_test, y_test


class StratifiedSplitTask(BaseSplitTask):
    """Meant for multilabel stratified slit"""

    def perform_split(self, X, y, test_size):
        """Perform the actual split using sklearn or skmultilearn"""

        # check if multilabel or multiclass dataset
        if self.is_multilabel:
            X = X.reshape(X.shape[0], 1)  # it needs this reshape for the split to work

            X_train, y_train, X_test, y_test = iterative_train_test_split(
                X, y, test_size=test_size
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y
            )

        return X_train, y_train, X_test, y_test
