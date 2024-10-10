import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
150 images of size 6800x7200 pixels with 3m resolution. The images were collected by Gaofen-2 over 60 different cities in China, between 2015 and 2016. 
The dataset, named as Gaofen Image Dataset with 15 categories (GID-15), is a land cover dataset.
"""


class GID15Dataset(SemanticSegmentationDataset):
    url = "https://captain-whu.github.io/GID15/"

    labels = ["no_data","Industrial land","Urban residential","Rural residential","Traffic land","Paddy field","Irrigated land","Dry cropland","Garden land","Arbor forest","Shrub land","Natural meadow","Artificial meadow","River","Lake","Pond"]
    color_mapping = [[0,0,0],[200,0,0],[250,0,150],[200,150,150],[250,150,150],[0,200,0],[150,250,0],[150,200,150],[200,0,200],[150,0,250],[150,150,250],[250,200,0],[200,200,0],[0,0,200],[0,150,200],[0,200,250]] 
    name = "GID-15"
    
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

        ids = os.listdir(os.path.join(data_dir[: data_dir.rfind(".")], "img_dir", data_dir[data_dir.rfind(".") :]))
        self.images = [os.path.join(data_dir[: data_dir.rfind(".")], "img_dir", data_dir[data_dir.rfind(".") :], image_id) for image_id in ids]
        self.masks = [
            os.path.join(data_dir[: data_dir.rfind(".")], "ann_dir", data_dir[data_dir.rfind(".") :], image_id[: image_id.rfind(".")] + "_15label.png") 
            for image_id in ids
        ]
        
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