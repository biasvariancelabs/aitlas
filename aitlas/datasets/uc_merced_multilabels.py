import os

import torchvision.transforms as transforms

from ..base import SplitableDataset
from ..utils import pil_loader, tiff_loader
from .schemas import UcMercedMultiLabelsDatasetSchema
import numpy as np


IMG_EXTENSIONS = [
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

CLASSES_TO_IDX = {
    "airplane": 0,
    "bare-soil": 1,
    "buildings": 2,
    "cars": 3,
    "chaparral": 4,
    "court": 5,
    "dock": 6,
    "field": 7,
    "grass": 8,
    "mobile-home": 9,
    "pavement": 10,
    "sand": 11,
    "sea": 12,
    "ship": 13,
    "tanks": 14,
    "trees": 15,
    "water": 16,
}


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions='.tif'):
    # read labels
    multi_hot_labels = {}
    with open(dir + '/multilabels.txt', 'rb') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.decode("utf-8")
            labels_list = line[line.find('\t') + 1:].split('\t')
            multi_hot_labels[line[:line.find('\t')]] = np.asarray(list((map(int, labels_list))))

    images = []
    dir = os.path.expanduser(dir + '/images')
    print(dir)
    # this ensures the image always have the same index numbers
    for root, _, fnames in sorted(os.walk(dir)):
        print(len(fnames))
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                multi_hot_label = multi_hot_labels[fname[:fname.find(extensions)]]
                item = (path, multi_hot_label)
                images.append(item)

    return images


class UcMercedMultiLabelsDataset(SplitableDataset):
    schema = UcMercedMultiLabelsDatasetSchema

    url = "https://drive.google.com/file/d/1DtKiauowCB0ykjFe8v0OVvT76rEfOk0v/view"

    def __init__(self, config):
        # now call the constuctor to validate the schema and split the data
        SplitableDataset.__init__(self, config)
        self.image_loader = tiff_loader
        self.data = make_dataset(self.config.root, CLASSES_TO_IDX)

    def load_transforms(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = self.image_loader(self.data[index][0])
        # apply transformations
        img = self.transforms(img)
        target = self.data[index][1]
        return img, target

    def __len__(self):
        return len(self.data)

    def labels(self):
        return list(CLASSES_TO_IDX.keys())
