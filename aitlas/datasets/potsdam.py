import numpy as np
import os
import pandas as pd

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
38 images of size 6000x6000 pixels with resolution of 0.05m. The dataset was created to provide a benchmark for 
urban semantic segmentation. Each patch contains a true ortophoto and a DSM. Images were acquired over Potsdam 
in Germany using photogrammetric digital airborne camera systems and Airborne Laser Scanning (lidar).
"""


class PotsdamDataset(SemanticSegmentationDataset):
    url = "https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx"

    labels = ["no-data","Impervious surface","Clutter","Car","Tree","Low vegetation","Building"]
    color_mapping = [[0,0,0],[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255],[0,0,255]] 
    name = "Potsdam"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "masks", image_id) for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count