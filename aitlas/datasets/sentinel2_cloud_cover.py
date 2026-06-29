import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import Sentinel2CloudCoverSchema

"""
This dataset was generated as part of a crowdsourcing competition. The dataset consists of Sentinel-2 
satellite imagery and corresponding cloudy labels stored as GeoTiffs. There are 22728 chips collected 
between 2018 and 2020.
"""


class Sentinel2CloudCoverSegmentationDataset(SemanticSegmentationDataset):
    url = "https://source.coop/repositories/radiantearth/cloud-cover-detection-challenge/access"

    labels = ["no clouds","clouds"]
    color_mapping = [[0,0,0],[255,255,255]] 
    name = "Sentinel-2 Cloud Cover Segmentation Dataset"
    schema = Sentinel2CloudCoverSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.bands_s2 = self.config.bands_s2

    def __getitem__(self, index):
        imageB02 = image_loader(self.images_B02[index])
        imageB03 = image_loader(self.images_B03[index])
        imageB04 = image_loader(self.images_B04[index])
        imageB08 = image_loader(self.images_B08[index])

        if self.selection == 'rgb':
            image = np.dstack((imageB02,imageB03,imageB04))/5000
        elif self.selection == 'all': # consider all 4 channels
            image = np.dstack((imageB02,imageB03,imageB04,imageB08))/5000
        elif self.selection == "bands":
            if self.bands_s2 == None:
                print("The config must contain a bands_s2 field with a list of chosen bands")
            idx = []
            s2_bands = ["B02","B03","B04","B08"] # available s2 bands in this dataset
            image = np.dstack((imageB02,imageB03,imageB04,imageB08))/5000
            for b in self.bands_s2:
                if b not in s2_bands:
                    print("The bands must be valid and available Sentinel-2 bands, i.e. one of B02, B03, B04, B08")
                    break
                else:
                    idx.append(s2_bands.index(b))
            image = image[:,:,idx]

        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        # the original dataset is split into train/test sets
        # csv_files created by BVLabs provide a 60/20/20 split stratified based on location and time
        if csv_file:
            ids = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    chip_id = row[0] 
                    location = row[1]  
                    datetime = row[2] 
                    original_split = row[3]
                    ids.append((chip_id, original_split))
            self.images_B02 = [os.path.join(data_dir, image_id[1], "features", image_id[0], 'B02.tif') for image_id in ids]
            self.images_B03 = [os.path.join(data_dir, image_id[1], "features", image_id[0], 'B03.tif') for image_id in ids]
            self.images_B04 = [os.path.join(data_dir, image_id[1], "features", image_id[0], 'B04.tif') for image_id in ids]
            self.images_B08 = [os.path.join(data_dir, image_id[1], "features", image_id[0], 'B08.tif') for image_id in ids]
            self.masks = [os.path.join(data_dir, image_id[1], "labels", image_id[0] + '.tif') for image_id in ids]

        else: # original split
            ids = os.listdir(os.path.join(data_dir, "features"))
            self.images_B02 = [os.path.join(data_dir, "features", image_id, 'B02.tif') for image_id in ids]
            self.images_B03 = [os.path.join(data_dir, "features", image_id, 'B03.tif') for image_id in ids]
            self.images_B04 = [os.path.join(data_dir, "features", image_id, 'B04.tif') for image_id in ids]
            self.images_B08 = [os.path.join(data_dir, "features", image_id, 'B08.tif') for image_id in ids]
            self.masks = [os.path.join(data_dir, "labels", image_id + '.tif') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    
    def __len__(self):
        return len(self.images_B02)