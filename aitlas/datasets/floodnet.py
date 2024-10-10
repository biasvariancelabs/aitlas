import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
The dataset was created to enhance disaster management and advance the damage assessment process for post-disaster scenarios. 
FloodNet provides 2343 high-resolution UAS images with detailed semantic annotations focusing on damage assessment after Hurricane Harvey.
"""


class FloodNetDataset(SemanticSegmentationDataset):
    url = ["https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH","https://drive.google.com/drive/folders/1g1r419bWBe4GEF-7si5DqWCjxiC8ErnY"]

    labels = ["background","building-flooded","building-non-flooded","road-flooded","road-non-flooded","water","tree","vehicle","pool","grass"]
    color_mapping = [[0,0,0],[255,51,51],[179,81,77],[161,161,0],[153,153,153],[0,255,255],[51,51,255],[255,102,255],[255,0,0],[51,255,51]] 
    name = "FloodNet"

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

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig