import numpy as np
import os
import pandas as pd

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader

"""
The Aeroscapes dataset was designed to enhance semantic understanding of urban scenes. The dataset comprises 3269 images acquired from 141 video sequences. 
The images have a size 1280x720 pixels and are in png format. 
"""


class AeroscapesDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/file/d/1WmXcm0IamIA0QPpyxRfWKnicxZByA60v/view"

    labels = ["Background","Person","Bike","Car","Drone","Boat","Animal","Obstacle","Construction","Vegetation","Road","Sky"]
    color_mapping = [[0,0,0],[192,128,128],[0,128,1],[128,128,128],[128,0,0],[1,0,128],[192,0,129],[191,0,0],[192,129,0],[0,64,1],[127,128,0],[0,128,129]] 
    name = "Aeroscapes"

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
        self.masks = [os.path.join(data_dir, "masks", image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count