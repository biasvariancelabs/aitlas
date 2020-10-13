import os

import torchvision.transforms as transforms

from ..base import SplitableDataset
from ..utils import pil_loader, tiff_loader
from .schemas import UcMercedDatasetSchema


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
    "agricultural": 0,
    "airplane": 1,
    "baseballdiamond": 2,
    "beach": 3,
    "buildings": 4,
    "chaparral": 5,
    "denseresidential": 6,
    "forest": 7,
    "freeway": 8,
    "golfcourse": 9,
    "harbor": 10,
    "intersection": 11,
    "mediumresidential": 12,
    "mobilehomepark": 13,
    "overpass": 14,
    "parkinglot": 15,
    "river": 16,
    "runway": 17,
    "sparseresidential": 18,
    "storagetanks": 19,
    "tenniscourt": 20,
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


class UcMercedDataset(SplitableDataset):
    schema = UcMercedDatasetSchema

    url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"

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
