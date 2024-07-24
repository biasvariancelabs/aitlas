import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
803 images captured over Thailand, Indonesia, and India. The dataset comes from the DeepGlobe Land Cover Classification 
Challenge which offers high-resolution sub-meter satellite imagery focusing on rural areas. All images contain RGB data 
collected from the DigitalGlobe Vivid+ dataset. Each satellite image is paired with a mask image for land cover annotation. 
The mask is an RGB image with 7 classes following the Anderson Classification.
"""


class DeepGlobeDataset(SemanticSegmentationDataset):
    url = "https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset"

    labels = ["urban-land", "agriculture-land", "rangeland", "forest-land", "water", "barren-land", "unknown"]
    color_mapping = [[0,255,255],[255,255,0],[255,0,255],[0,255,0],[0,0,255],[255,255,255],[0,0,0]] 
    name = "DeepGlobe"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index]) 
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j,0] == 0 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 0
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 1
                if mask[i,j,0] == 0 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 2
                if mask[i,j,0] == 255 and mask[i,j,1] == 0 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 3
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 4
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 255:
                    single_band_mask[i,j] = 5
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 6
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [
            os.path.join(data_dir, "masks", image_id[: image_id.rfind("_")] + "_mask.png")
            for image_id in ids
        ]
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count_float = label_count["Number of pixels"].astype(float)
        return label_count_float.to_frame()