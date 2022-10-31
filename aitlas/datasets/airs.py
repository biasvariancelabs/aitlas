import numpy as np

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
This dataset contains 1171 aerial images, along with their respective maps.
They are 1500 x 1500 in dimension and are in .tiff format
"""


class AIRSDataset(SemanticSegmentationDataset):
    url = "https://www.airs-dataset.com/"

    labels = ["Background", "Roof"]
    # Color mapping for the labels
    color_mapping = [[0, 0, 0], [200, 200, 200]]
    name = "AIRS"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])[:, :, 1]
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)
