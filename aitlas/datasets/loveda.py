import numpy as np
import os
import pandas as pd

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
5987 images of size 1024x1024 pixels with 0.03m resolution. The Remote Sensing Land-Cover dataset for Domain Adaptive 
Semantic Segmentation (LoveDA) was constructed using images obtained from Nanjing, Changzhou and Wuhan. Images come 
from Google Earth and were collected in 2016.
"""


class LoveDADataset(SemanticSegmentationDataset):
    url = "https://zenodo.org/records/5706578"

    labels = ["no-data","Background","Buildings","Road","Water","Barren","Forest","Agricultural"]
    color_mapping = [[0,0,0],[255,255,255],[255,0,0],[255,255,0],[0,0,255],[159,129,183],[0,255,0],[255,195,128]] 
    name = "LoveDA"

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