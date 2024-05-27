import numpy as np
import os
import pandas as pd

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
The WHU-Mix dataset is a diverse, large-scale, and high-quality dataset that aims to better simulate the situation of practical 
building extraction. It contains 51445 images.
"""


class WHUMixDataset(SemanticSegmentationDataset):
    url = "http://gpcv.whu.edu.cn/data/whu-mix(raster)/whu_mix%20(raster).html"

    labels = ["Background","Building"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "WHUMix"

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

        ids = os.listdir(os.path.join(data_dir[: data_dir.rfind('.')], "img_dir", data_dir[data_dir.rfind('.') :]))
        self.images = [os.path.join(data_dir[: data_dir.rfind('.')], "img_dir", data_dir[data_dir.rfind('.') :], image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir[: data_dir.rfind('.')], "ann_dir", data_dir[data_dir.rfind('.') :], image_id[: image_id.rfind('.tif')] + '.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count