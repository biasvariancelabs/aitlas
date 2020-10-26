import logging
import os
from ..base import BaseTask, BaseModel
from .schemas import SplitTaskSchema
from torch.utils.data import random_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SplitTask(BaseTask):
    schema = SplitTaskSchema

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

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        self.images = self.load_images(self.config.root)
        self.split()
        logging.info("And that's it!")

    def split(self):
        if not self.is_split_valid():
            raise ValueError(
                "The defined split is invalid. The sum should be equal to 100."
            )
        # split the dataset
        self.make_splits()

    def has_val(self):
        return self.config.split.val

    def is_split_valid(self):
        res = self.config.split.train.ratio + self.config.split.test.ratio
        if self.has_val():
            res += self.config.split.val.ratio
        return res == 100

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

        # save the splits

        self.train_indices = result[0]
        self.test_indices = result[1]

        if self.has_val():
            self.val_indices = result[2]

        self.save_split(result[0], self.config.split.train.file)
        self.save_split(result[1], self.config.split.test.file)
        if self.has_val():
            self.save_split(result[2], self.config.split.val.file)

    def save_split(self, indices, file):
        with open(file, "w") as f:
            for ind in indices:
                f.write("{},{}\n".format(self.images[ind][0], self.images[ind][1]))
            f.close()

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
        classes = [item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item))]

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


