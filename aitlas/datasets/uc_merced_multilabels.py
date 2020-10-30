import os

import numpy as np

from ..base import BaseDataset
from ..utils import tiff_loader
from .schemas import PascalVOCMultilabelDatasetSchema


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


class UcMercedMultiLabelsDataset(BaseDataset):

    schema = PascalVOCMultilabelDatasetSchema

    url = "https://drive.google.com/file/d/1DtKiauowCB0ykjFe8v0OVvT76rEfOk0v/view"

    classes_to_idx = CLASSES_TO_IDX

    def __init__(self, config):
        # now call the constuctor to validate the schema and split the data
        BaseDataset.__init__(self, config)

        self.image_loader = tiff_loader
        self.data = self.make_dataset(self.config.root)

    def make_dataset(self, dir, extensions=".tif"):
        # read labels
        multi_hot_labels = {}
        with open(dir + "/multilabels.txt", "rb") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.decode("utf-8")
                labels_list = line[line.find("\t") + 1 :].split("\t")
                multi_hot_labels[line[: line.find("\t")]] = np.asarray(
                    list((map(float, labels_list)))
                )

        images = []
        dir = os.path.expanduser(dir + "/images")
        # this ensures the image always have the same index numbers
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                multi_hot_label = multi_hot_labels[fname[: fname.find(extensions)]]
                item = (path, multi_hot_label)
                images.append(item)

        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = self.image_loader(self.data[index][0])
        img = self.transform(img)
        target = self.data[index][1]
        return img, target

    def __len__(self):
        return len(self.data)

    def labels(self):
        return list(self.classes_to_idx.keys())
