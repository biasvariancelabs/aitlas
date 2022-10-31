import os
import numpy as np

from ..utils import image_loader
from .semantic_segmentation import SemanticSegmentationDataset

"""
For the CamVid dataset the mask is in one file, each label is color coded.
"""


class CamVidDataset(SemanticSegmentationDataset):
    url = "https://github.com/alexgkendall/SegNet-Tutorial"

    labels = [
        "sky",
        "building",
        "column_pole",
        "road",
        "sidewalk",
        "tree",
        "sign",
        "fence",
        "car",
        "pedestrian",
        "byciclist",
        "void",
    ]
    color_mapping = [[255, 127, 127], [255, 191, 127], [255, 255, 127], [191, 255, 127], [127, 255, 127],
                     [127, 255, 191], [127, 255, 255], [127, 191, 255], [127, 127, 255],
                     [191, 127, 255], [255, 127, 255], [255, 127, 191]]
    name = "CamVid"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], False)
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "masks", image_id) for image_id in ids]
