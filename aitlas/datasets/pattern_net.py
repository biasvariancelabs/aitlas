import os

import torchvision.transforms as transforms

from ..base import SplitableDataset
from ..utils import pil_loader, tiff_loader
from .schemas import PatternNetDatasetSchema


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
    "baseball_field": 1,
    "basketball_court": 2,
    "beach": 3,
    "bridge": 4,
    "cemetery": 5,
    "chaparral": 6,
    "christmas_tree_farm": 7,
    "closed_road": 8,
    "coastal_mansion": 9,
    "crosswalk": 10,
    "dense_residential": 11,
    "ferry_terminal": 12,
    "football_field": 13,
    "forest": 14,
    "freeway": 15,
    "golf_course": 16,
    "harbor": 17,
    "intersection": 18,
    "mobile_home_park": 19,
    "nursing_home": 20,
    "oil_gas_field": 21,
    "oil_well": 22,
    "overpass": 23,
    "parking_lot": 24,
    "parking_space": 25,
    "railway": 26,
    "river": 27,
    "runway": 28,
    "runway_marking": 29,
    "shipping_yard": 30,
    "solar_panel": 31,
    "sparse_residential": 32,
    "storage_tank": 33,
    "swimming_pool": 34,
    "tennis_court": 35,
    "transformer_station": 36,
    "wastewater_treatment_plant": 37,
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


def make_dataset(dir, class_to_idx, extensions='.jpg'):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        # this ensures the image always have the same index numbers
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class PatternNetDataset(SplitableDataset):
    schema = PatternNetDatasetSchema

    url = "https://sites.google.com/view/zhouwx/dataset"

    def __init__(self, config):
        # now call the constuctor to validate the schema and split the data
        SplitableDataset.__init__(self, config)
        self.image_loader = pil_loader
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
