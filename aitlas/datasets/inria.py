from .semantic_segmentation import SemanticSegmentationDataset

"""
The training set contains 180 color image tiles of size 5000×5000, covering a surface of 1500 m × 1500 m each
(at a 30 cm resolution). The format is GeoTIFF (TIFF with georeferencing, but the images can be used as any other TIFF).
The reference data is in a different folder and the file names correspond exactly to those of the color images.
In the case of the reference data, the tiles are single-channel images
with values 255 for the building class and 0 for the not building class.
Use function split_images from utils to split the images and the masks in smaller patches
"""


class InriaDataset(SemanticSegmentationDataset):
    url = "https://project.inria.fr/aerialimagelabeling/"

    labels = ["Background", "Buildings"]
    # Color mapping for the labels
    color_mapping = [[0, 0, 0], [255, 255, 255]]
    name = "Inria"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
