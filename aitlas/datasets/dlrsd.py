import numpy as np
import os
import pandas as pd
import csv

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
DLRSD (Dense Labeling Remote Sensing Dataset) is a dense labeling dataset that can 
be used for multi-label tasks such as remote sensing image retrieval (RSIR) and 
classification, and other pixel-based tasks like semantic segmentation. It has 21 
broad categories with 100 images per class, and 17 segmentation classes.
"""


class DLRSDDataset(SemanticSegmentationDataset):
    url = "https://sites.google.com/view/zhouwx/dataset#h.p_hQS2jYeaFpV0"

    labels = ["no data","airplane","bare soil","buildings","cars","chaparral","court","dock","field","grass","mobile home","pavement","sand","sea","ship","tanks","trees","water"]
    color_mapping = [[0,0,0],[165,202,239],[128,130,0],[1,0,126],[254,2,0],[0,128,0],[126,0,1],[251,229,230],[159,158,163],[2,128,129],[87,85,249],[249,253,4],[253,190,8],[0,2,249],[251,0,190],[126,0,133],[1,252,1],[4,251,253]] 
    name = "DLRSD"
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
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "Images"))
        
        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'Images/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        self.images = [os.path.join(data_dir, "Images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "Labels", image_id[: image_id.rfind('.tif')]+'.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count