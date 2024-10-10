import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
420 images of size 4096x2160 pixels. Images come from videos collected by UAV over 30 different places. The dataset was designed 
for semantic segmentation in complex urban scenes, featuring on both static and moving object recognition.
"""


class UAVidDataset(SemanticSegmentationDataset):
    url = "https://uavid.nl/"

    labels = ["clutter","building","road","tree","low vegetation","moving car","static car","human"]
    color_mapping = [[128,0,0],[128,64,128],[0,128,0],[128,128,0],[64,0,128],[192,0,192],[64,64,0],[0,0,0]] 
    name = "UAVid"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index],False)
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