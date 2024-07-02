import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch
from matplotlib.patches import Patch

from .schemas import CloudDatasets_AI4QCSchema
from ..base import BaseDataset
from torch.utils.data import DataLoader, Dataset
from ..utils import tiff_loader
from skimage.transform import resize
from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader


def interp_band(bands, img_shape=[256, 256]):
    
    bands_interp = np.zeros(img_shape).astype(np.float32)
    bands_interp = resize(bands/10000, img_shape, mode="reflect")*10000 #10000 because of the reflectance mode (initial values are DN)

    return bands_interp

'''
Reference cloud data over NASA GSFC for various cloud conditions from September 2017 to November 2018. 
The reference dataset is used for validating cloud masking algorithms for moderate spatial resolution satellite data, 
namely Sentinel-2. The dataset is a smaller version of the GSFC dataset, where S2 L1C images were added, the cloud masks 
reclassified and rasterized.
'''
    
class GSFC_AI4QCDataset(SemanticSegmentationDataset):

    url = "https://zenodo.org/records/11120856"

    labels = ["clear","thick cloud","thin cloud","cloud shadow"]
    color_mapping = [[255,255,255],[0,0,255],[0,255,255],[128,128,128]]
    name = "GSFC_AI4QC"
    schema = CloudDatasets_AI4QCSchema
    
    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection

    def __getitem__(self, index):
        mask = np.abs(image_loader(self.masks[index],False))
        masks = [(mask == v) for v, label in enumerate(self.labels)]
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

            #normalize
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
        self.masks = [os.path.join(data_dir, "masks", image_id) for image_id in ids]

    #dataloader adapted to the varying sizes of the image/mask pairs
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

    def show_image_13_bands_data(self, index, show_title=False):
        img, mask = self[index]
        img = img[:,:,1].astype(np.float64) #plots the band B02
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
