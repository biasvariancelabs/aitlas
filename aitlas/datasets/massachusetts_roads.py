from .semantic_segmentation import SemanticSegmentationDataset

"""
This dataset contains 1171 aerial images, along with their respective maps.
They are 1500 x 1500 in dimension and are in .tiff format
"""


class MassachusettsRoadsDataset(SemanticSegmentationDataset):
    url = "https://www.cs.toronto.edu/~vmnih/data/"

    labels = ["Background", "Roads"]
    # Color mapping for the labels
    color_mapping = [[0, 0, 0], [200, 200, 200]]
    name = "Massachusetts Roads"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
