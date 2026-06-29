import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The Kaggle Satellite Buildings Semantic Segmentation dataset contains 6038 images and masks
for buildings detection through semantic segmentation.
"""


class SatelliteBuildingsDataset(SemanticSegmentationDataset):
    url = "https://www.kaggle.com/datasets/hyyyrwang/buildings-dataset/data"

    labels = ["background","building"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "Satellite Buildings"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j] > 0:
                    single_band_mask[i,j] = 1
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        # the dataset originally comes with a train/test split of 6023/8
        # csv_files are provided by BVLabs with a 60%/20%/20% train/val/test split of 3619/1206/1206
        if csv_file:
            ids = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    img_id = row[0].split('/')[1]
                    ids.append((img_id))

        else: # original split
            ids = os.listdir(os.path.join(data_dir, "src"))

        self.images = [os.path.join(data_dir, "src", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "label", image_id) for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.images)):  #because images have different sizes
            mask = self[i][1]
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count