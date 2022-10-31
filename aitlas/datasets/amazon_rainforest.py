import os
import numpy as np

from ..utils import image_loader
from .semantic_segmentation import SemanticSegmentationDataset

"""
The Amazon Rainforest dataset for semantic segmentation
contains GeoTIFF images with 512x512 pixels and associated PNG masks
(forest indicated in white and non-forest in black color)
"""


class AmazonRainforestDataset(SemanticSegmentationDataset):
    url = "https://zenodo.org/record/3233081#.YTYm_44zaUk"

    labels = ["Background", "Forest"]
    # Color mapping for the labels
    color_mapping = [[0, 0, 0], [0, 255, 0]]
    name = "Amazon Rainforest"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], True) / 255
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [
            os.path.join(data_dir, "masks", image_id[: image_id.rfind(".")] + ".png")
            for image_id in ids
        ]
