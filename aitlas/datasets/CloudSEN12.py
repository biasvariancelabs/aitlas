import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
import torch
from PIL import Image

from .schemas import CloudDatasets_AI4QCSchema
from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from ..base import BaseDataset
from ..base import BaseTransforms


def interp_band(bands, img_shape=[509, 509]):
    
    bands_interp = np.zeros([bands.shape[2]] + img_shape).astype(np.float32)

    for i in range(bands.shape[2]):
        bands_interp[i] = resize(bands[i]/10000, img_shape, mode="reflect")*10000 #10000 because of the reflectance mode (initial values are DN)

    return bands_interp.transpose(1,2,0)

"""
Mutli-temporal global dataset created to foster research in cloud and cloud shadow detection. The dataset contains three 
type of annotaions: high-quality (2000 ROI), scribble (2000 ROI) and no-annotation (5880 ROI). Each ROI contains 5 patches 
with different cloud covers: clear (0%), low-cloudy (1% - 25%), almost clear (25% - 45%), mid-cloudy (45% - 65%), cloudy (65% >).
"""

class CloudSEN12Dataset(SemanticSegmentationDataset):
    url = "https://www.scidb.cn/en/detail?dataSetId=f72d622ff4ea4fa18070456a98222b1a"

    labels = ["clear","thick cloud","thin cloud","cloud shadow"]
    color_mapping = [[255,255,255],[0,0,255],[0,255,255],[0,0,0]] 
    name = "CloudSEN12"
    schema = CloudDatasets_AI4QCSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        
        if self.selection == "rgb":
            image = np.array(image[:, :, [3, 2, 1]])
            #transpose and normalize
            image = torch.tensor(image.transpose(2,0,1), dtype=torch.float32) / 10000 
            
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask

        elif self.selection == "all":
            bands10 = image[:, :, [1, 2, 3, 7]]
            bands20 = image[:,:, [4, 5, 6, 8, 11, 12]]
            bands20 = interp_band(bands20)
            bands60 = image[:,:, [0, 9, 10]]
            bands60 = interp_band(bands60)
            bands10 = bands10.astype(np.float32)
            bands20 = bands20.astype(np.float32)
            bands60 = bands60.astype(np.float32)
            bands = np.dstack((bands10,bands20,bands60))
            #transpose and normalize
            bands = torch.tensor(bands.transpose(2,0,1), dtype=torch.float32) / 10000

            if self.transform:
                bands = self.transform(bands)
            if self.target_transform:
                mask = self.target_transform(mask)    

            return bands, mask
        
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

    def show_image(self, index, show_title=False):
        img, mask = self[index]
        #img = img[:,:,1].astype(np.float64) #plots the band B02
        img = img[:,:,[2,1,0]]
        #img = torch.tensor(img, dtype=torch.float32) / 10000
        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig