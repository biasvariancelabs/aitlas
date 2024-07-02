import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from ..base import BaseDataset
from torch.utils.data import DataLoader, Dataset
from ..utils import tiff_loader
from matplotlib.patches import Patch
from skimage.transform import resize
from .schemas import CloudDatasets_AI4QCSchema


def interp_band(bands, img_shape=[256, 256]):
    
    bands_interp = np.zeros(img_shape).astype(np.float32)
    bands_interp = resize(bands/10000, img_shape, mode="reflect")*10000 #10000 because of the reflectance mode (initial values are DN)

    return bands_interp

"""
The S2 Hollstein dataset is a globally distributed database of manually labeled Sentinel-2 spectra of clouds. 
It contains 58 cloud masks.
"""


class HollsteinDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/1nT-Jr_0Qmr9BIU8kgHNY1e-HL3cdH4KS?usp=sharing"

    labels = ["no_data","clear","water","shadow","cirrus","cloud","snow"] #add the no_data class to start at 0
    color_mapping = [[0,0,0],[255,255,255],[0,0,128],[128,128,128],[0,255,255],[0,0,255],[192,192,192]] 
    name = "Hollstein"
    schema = CloudDatasets_AI4QCSchema
    
    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection

    def __getitem__(self, index):
        mask = image_loader(self.masks[index],False)
        masks = [(mask == v*10) for v, label in enumerate(self.labels)] #the label values are actually 10, 20, 30, 40, 50, 60
        mask = np.stack(masks, axis=-1).astype("float32")
        
        if self.selection == "rgb":
            image = image_loader(self.images[index])
            return self.apply_transformations(image, mask)

        elif self.selection == "all":
            imageB01 = tiff_loader(self.imagesB01[index])
            imageB01 = interp_band(imageB01)
            imageB02 = tiff_loader(self.imagesB02[index])
            imageB02 = interp_band(imageB02)
            imageB03 = tiff_loader(self.imagesB03[index])
            imageB03 = interp_band(imageB03)
            imageB04 = tiff_loader(self.imagesB04[index])
            imageB04 = interp_band(imageB04)
            imageB05 = tiff_loader(self.imagesB05[index])
            imageB05 = interp_band(imageB05)
            imageB06 = tiff_loader(self.imagesB06[index])
            imageB06 = interp_band(imageB06)
            imageB07 = tiff_loader(self.imagesB07[index])
            imageB07 = interp_band(imageB07)
            imageB08 = tiff_loader(self.imagesB08[index])
            imageB08 = interp_band(imageB08)
            imageB8A = tiff_loader(self.imagesB8A[index])
            imageB8A = interp_band(imageB8A)
            imageB09 = tiff_loader(self.imagesB09[index])
            imageB09 = interp_band(imageB09)
            imageB10 = tiff_loader(self.imagesB10[index])
            imageB10 = interp_band(imageB10)
            imageB11 = tiff_loader(self.imagesB11[index])
            imageB11 = interp_band(imageB11)
            imageB12 = tiff_loader(self.imagesB12[index])
            imageB12 = interp_band(imageB12)
            
            image = np.array([imageB01,imageB02,imageB03,imageB04, imageB05,imageB06,imageB07,imageB08,imageB8A,imageB09,imageB10,imageB11,imageB12])
            image = image.astype(np.float32)
            image = torch.tensor(image, dtype=torch.float32) / 10000

            if self.transform:
                image, mask = self.transform(image)
            if self.target_transform:
               mask = self.target_transform(mask)

            return image, mask

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "images"))
        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.imagesB01 = [os.path.join(data_dir, "B01", image_id[: image_id.rfind('.tif')]+'_B01.tif') for image_id in ids]
        self.imagesB02 = [os.path.join(data_dir, "B02", image_id[: image_id.rfind('.tif')]+'_B02.tif') for image_id in ids]
        self.imagesB03 = [os.path.join(data_dir, "B03", image_id[: image_id.rfind('.tif')]+'_B03.tif') for image_id in ids]
        self.imagesB04 = [os.path.join(data_dir, "B04", image_id[: image_id.rfind('.tif')]+'_B04.tif') for image_id in ids]
        self.imagesB05 = [os.path.join(data_dir, "B05", image_id[: image_id.rfind('.tif')]+'_B05.tif') for image_id in ids]
        self.imagesB06 = [os.path.join(data_dir, "B06", image_id[: image_id.rfind('.tif')]+'_B06.tif') for image_id in ids]
        self.imagesB07 = [os.path.join(data_dir, "B07", image_id[: image_id.rfind('.tif')]+'_B07.tif') for image_id in ids]
        self.imagesB08 = [os.path.join(data_dir, "B08", image_id[: image_id.rfind('.tif')]+'_B08.tif') for image_id in ids]
        self.imagesB8A = [os.path.join(data_dir, "B8A", image_id[: image_id.rfind('.tif')]+'_B8A.tif') for image_id in ids]
        self.imagesB09 = [os.path.join(data_dir, "B09", image_id[: image_id.rfind('.tif')]+'_B09.tif') for image_id in ids]
        self.imagesB10 = [os.path.join(data_dir, "B10", image_id[: image_id.rfind('.tif')]+'_B10.tif') for image_id in ids]
        self.imagesB11 = [os.path.join(data_dir, "B11", image_id[: image_id.rfind('.tif')]+'_B11.tif') for image_id in ids]
        self.imagesB12 = [os.path.join(data_dir, "B12", image_id[: image_id.rfind('.tif')]+'_B12.tif') for image_id in ids]
        self.masks = [os.path.join(data_dir, "masks_before_reclassification", 'mask_'+ image_id) for image_id in ids]

    def dataloader_varying_sizes(self, indices):
        """Create and return a dataloader for the dataset"""
        return torch.utils.data.Subset(
            self,
            indices
        )

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in range(len(self)):
            for image, mask in self.dataloader_varying_sizes([i]):
                for index, label in enumerate(self.labels):
                    label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count