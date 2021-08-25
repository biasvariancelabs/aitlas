import logging
import os

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import random_split

from ..base import BaseModel, BaseTask
from ..utils import load_voc_format_dataset
from .schemas import RandomSplitTaskSchema, SplitTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseSplitTask(BaseTask):
    """Base task meant to split dataset"""

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        self.images = self.load_images(self.config.root)
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

    def get_image_row_string(self, image):
        return "{},{}\n".format(image[0], image[1])

    def get_image_row(self, x):
        return self.get_image_row_string(self.images[x])

    def save_split(self, data, file):
        with open(file, "w") as f:
            for d in data:
                f.write(self.get_image_row(d))
            f.close()

    def make_splits(self):
        raise NotImplementedError

    def load_images(self, dir, extensions=None):
        raise NotImplementedError


class RandomSplitTask(BaseSplitTask):
    """Randomly split a folder containing images"""

    schema = RandomSplitTaskSchema

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

    def make_splits(self):
        size = len(self.images)
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

        # save the splits
        self.save_split(result[0], self.config.split.train.file)
        self.save_split(result[1], self.config.split.test.file)

        if self.has_val():
            self.save_split(result[2], self.config.split.val.file)

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

    def load_images(self, dir, extensions=None):
        if not extensions:
            extensions = self.extensions

        images = []
        dir = os.path.expanduser(dir)
        classes = [
            item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item))
        ]

        for target in classes:
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, target)
                        images.append(item)

        return images


class StratifiedSplitTask(BaseSplitTask):
    """Meant for multilabel stratified slit"""

    schema = RandomSplitTaskSchema

    def load_images(self, dir, extensions=None):
        """ this fill transform images in the format ["path",[labels]]"""
        return load_voc_format_dataset(dir)

    def get_image_row_string(self, image):
        return "{}\n".format(image)

    def get_image_row(self, x):
        return self.get_image_row_string(x[0])

    def make_splits(self):
        X = np.array([x[0] for x in self.images])
        y = np.array([x[1] for x in self.images])

        X = X.reshape(X.shape[0], 1)  # it needs this reshape for the split to work

        test_size = float(self.config.split.test.ratio / 100)

        X_train, y_train, X_test, y_test = iterative_train_test_split(
            X, y, test_size=test_size
        )

        # save the splits
        self.save_split(X_train, self.config.split.train.file)
        self.save_split(X_test, self.config.split.test.file)

        if self.has_val():
            val_size = float(
                self.config.split.val.ratio
                / (self.config.split.val.ratio + self.config.split.train.ratio)
            )
            X_train, y_train, X_val, y_val = iterative_train_test_split(
                X_train, y_train, test_size=val_size
            )

            # save split
            self.save_split(X_val, self.config.split.val.file)
