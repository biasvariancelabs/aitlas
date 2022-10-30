import numpy as np

from ..utils import image_loader
from .semantic_segmentation import SemanticSegmentationDataset

"""
This dataset contains 1171 aerial images, along with their respective maps.
They are 1500 x 1500 in dimension and are in .tiff format
"""


class AIRSDataset(SemanticSegmentationDataset):
    url = "https://www.airs-dataset.com/"

    labels = ["Background", "Roof"]
    # Color mapping for the labels
    color_mapping = [[0, 0, 0], [255, 255, 255]]
    name = "AIRS"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
