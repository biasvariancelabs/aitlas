import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch

from ..base import BaseDataset
from torch.utils.data import DataLoader, Dataset
from ..utils import image_loader
from ..utils import tiff_loader
from matplotlib.patches import Patch
from skimage.transform import resize
from .semantic_segmentation import SemanticSegmentationDataset
from .schemas import CloudDatasets_AI4QCSchema


def interp_band(bands, img_shape=[512, 512]):
    
    bands_interp = np.zeros(img_shape).astype(np.float32)
    bands_interp = resize(bands, img_shape, mode="reflect") # values are reflectances (between 0 and 1)

    return bands_interp       

"""
KappaSet is a cloud mask dataset of 9251 labeled subscenes distributed over the full globe for the whole year 2020 
(Winter, Spring, Summer and Autumn products). To each label file is associated a Sentinel-2 L1C image.
"""

class KappaSetDataset(SemanticSegmentationDataset):

    url = "https://drive.google.com/drive/folders/1H18RIitlVvhmlY63_lj_BrtYPwzN8tgr?usp=sharing"
    
    labels = ["UNDEFINED","CLEAR","CLOUD SHADOW","SEMI TRANSPARENT CLOUD","CLOUD","MISSING"]
    color_mapping = [[0,0,0],[255,255,255],[128,128,128],[0,255,255],[0,0,255],[128,0,0]] 
    name = "KappaSet"
    schema = CloudDatasets_AI4QCSchema
    
    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection

    def __getitem__(self, index):
        mask = image_loader(self.masks[index],False)
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        
        if self.selection == "rgb":
            imageB02 = image_loader(self.imagesB02[index])
            imageB03 = image_loader(self.imagesB03[index])
            imageB04 = image_loader(self.imagesB04[index])
            image = np.array([imageB02, imageB03, imageB04])
            image = image.astype(np.float32)
            
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            return image, mask

        elif self.selection == "all":
            imageB01 = image_loader(self.imagesB01[index])
            imageB01 = interp_band(imageB01)
            imageB02 = image_loader(self.imagesB02[index])
            imageB02 = interp_band(imageB02)
            imageB03 = image_loader(self.imagesB03[index])
            imageB03 = interp_band(imageB03)
            imageB04 = image_loader(self.imagesB04[index])
            imageB04 = interp_band(imageB04)
            imageB05 = image_loader(self.imagesB05[index])
            imageB05 = interp_band(imageB05)
            imageB06 = image_loader(self.imagesB06[index])
            imageB06 = interp_band(imageB06)
            imageB07 = image_loader(self.imagesB07[index])
            imageB07 = interp_band(imageB07)
            imageB08 = image_loader(self.imagesB08[index])
            imageB08 = interp_band(imageB08)
            imageB8A = image_loader(self.imagesB8A[index])
            imageB8A = interp_band(imageB8A)
            imageB09 = image_loader(self.imagesB09[index])
            imageB09 = interp_band(imageB09)
            imageB10 = image_loader(self.imagesB10[index])
            imageB10 = interp_band(imageB10)
            imageB11 = image_loader(self.imagesB11[index])
            imageB11 = interp_band(imageB11)
            imageB12 = image_loader(self.imagesB12[index])
            imageB12 = interp_band(imageB12)
            
            image = np.array([imageB01,imageB02,imageB03,imageB04, imageB05,imageB06,imageB07,imageB08,imageB8A,imageB09,imageB10,imageB11,imageB12])
            image = image.astype(np.float32)

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
               mask = self.target_transform(mask)

            return image, mask

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        ids = os.listdir(os.path.join(data_dir, "Sentinel_2_B02/B02"))
        self.images = [os.path.join(data_dir, "Sentinel_2_B02/B02", image_id) for image_id in ids]
        self.imagesB01 = [os.path.join(data_dir, "Sentinel_2_B01/B01", image_id[: image_id.rfind('_B02.tif')]+'_B01.tif') for image_id in ids]
        self.imagesB02 = [os.path.join(data_dir, "Sentinel_2_B02/B02", image_id) for image_id in ids]
        self.imagesB03 = [os.path.join(data_dir, "Sentinel_2_B03/B03", image_id[: image_id.rfind('_B02.tif')]+'_B03.tif') for image_id in ids]
        self.imagesB04 = [os.path.join(data_dir, "Sentinel_2_B04/B04", image_id[: image_id.rfind('_B02.tif')]+'_B04.tif') for image_id in ids]
        self.imagesB05 = [os.path.join(data_dir, "Sentinel_2_B05/B05", image_id[: image_id.rfind('_B02.tif')]+'_B05.tif') for image_id in ids]
        self.imagesB06 = [os.path.join(data_dir, "Sentinel_2_B06/B06", image_id[: image_id.rfind('_B02.tif')]+'_B06.tif') for image_id in ids]
        self.imagesB07 = [os.path.join(data_dir, "Sentinel_2_B07/B07", image_id[: image_id.rfind('_B02.tif')]+'_B07.tif') for image_id in ids]
        self.imagesB08 = [os.path.join(data_dir, "Sentinel_2_B08/B08", image_id[: image_id.rfind('_B02.tif')]+'_B08.tif') for image_id in ids]
        self.imagesB8A = [os.path.join(data_dir, "Sentinel_2_B8A/B8A", image_id[: image_id.rfind('_B02.tif')]+'_B8A.tif') for image_id in ids]
        self.imagesB09 = [os.path.join(data_dir, "Sentinel_2_B09/B09", image_id[: image_id.rfind('_B02.tif')]+'_B09.tif') for image_id in ids]
        self.imagesB10 = [os.path.join(data_dir, "Sentinel_2_B10/B10", image_id[: image_id.rfind('_B02.tif')]+'_B10.tif') for image_id in ids]
        self.imagesB11 = [os.path.join(data_dir, "Sentinel_2_B11/B11", image_id[: image_id.rfind('_B02.tif')]+'_B11.tif') for image_id in ids]
        self.imagesB12 = [os.path.join(data_dir, "Sentinel_2_B12/B12", image_id[: image_id.rfind('_B02.tif')]+'_B12.tif') for image_id in ids]
        self.masks = [os.path.join(data_dir, "old_labels", image_id[: image_id.rfind('_B02')] +'_label.tif') for image_id in ids]
        
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
        img = img.transpose(2,1,0)
        img = img[:,:,1].astype(np.float32) #plots the band B02
        #img = img[:,:,[2,1,0]]
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