import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv
import math
from concurrent.futures import ThreadPoolExecutor

from ..utils import image_loader
from .schemas import SEN12MSSchema
from ..base import BaseDataset

"""
The SEN12MS dataset contains 180662 patch triplets of corresponding Sentinel-1 dual-pol SAR data, 
Sentinel-2 multi-spectral images, and MODIS-derived land cover maps.
"""

class SEN12MSDataset(BaseDataset):
    url = "https://dataserv.ub.tum.de/s/m1474000"

    labels = ["no-data","Evergreen needleleaf forest","Evergreen broadleaf forest","Deciduous needleleaf forest",
              "Deciduous broadleaf forest","Mixed forest","Closed shrublands","Open shrublands","Woody savannas",
              "Savannas","Grasslands","Permanent wetlands","Croplands","Urban and built-up","Cropland/Natural vegetation mosaic",
              "Snow and ice","Barren or sparsely vegetated","Water"]
    color_mapping = [[255,255,255],[252,132,255],[253,0,1],[0,255,4],[2,0,255],[253,255,0],[1,255,255],[253,0,254],[123,2,1],
                     [122,255,255],[0,0,128],[255,255,126],[128,0,251],[253,128,0],[1,130,252],[126,255,0],[133,1,128],
                     [128,254,128]] 
    name = "SEN12MS"
    schema = SEN12MSSchema

    def __init__(self, config):
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.csv_file = self.config.csv_file
        self.bands_s2 = self.config.bands_s2
        self.bands_s1 = self.config.bands_s1
        
        # Pre-compute band indices once during initialization
        self.s2_idx = []
        if self.selection == "bands" and self.bands_s2:
            s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
            self.s2_idx = [s2_bands.index(b) for b in self.bands_s2 if b in s2_bands]
            
        self.s1_idx = []
        if self.selection == "bands" and self.bands_s1:
            s1_bands = ["VV","VH"]
            self.s1_idx = [s1_bands.index(b) for b in self.bands_s1 if b in s1_bands]

        self.num_labels = len(self.labels)

        self.s1images, self.s2images, self.masks = self.load_dataset(self.data_dir, self.csv_file)

    def _process_s2(self, s2image):
        if self.selection == 'rgb':
            s2image = s2image[:, :, 1:4]
        elif self.selection == "bands": 
            s2image = s2image[:, :, self.s2_idx]
            
        img_min, img_max = s2image.min(), s2image.max()
        s2image = (s2image - img_min) / (img_max - img_min + 1e-8)
        
        if self.transform:
            s2image = self.transform(s2image)
        return s2image

    def _process_s1(self, s1image):
        if self.selection == 'rgb':
            ratio = s1image[:, :, 0] / (s1image[:, :, 1] + 1e-8)
            s1image = np.dstack((s1image[:, :, 0], s1image[:, :, 1], ratio)) 
        elif self.selection == 'bands': 
            s1image = s1image[:, :, self.s1_idx]
            
        img_min, img_max = s1image.min(), s1image.max()
        s1image = (s1image - img_min) / (img_max - img_min + 1e-8)
        
        if self.transform:
            s1image = self.transform(s1image)
        return s1image

    def __getitem__(self, index):
        # Fetch appropriate images based on configuration
        if self.imagery == 's2':
            image = self._process_s2(image_loader(self.s2images[index]))
        elif self.imagery == 's1':
            image = self._process_s1(image_loader(self.s1images[index]))
        elif self.imagery == 'all':
            s1image = self._process_s1(image_loader(self.s1images[index]))
            s2image = self._process_s2(image_loader(self.s2images[index]))
            image = np.dstack((s1image, s2image))

        # Instant C-level broadcasting replaces Python list comprehension for mask processing, eliminating Python overhead
        mask = image_loader(self.masks[index])[:, :, 0]
        mask = (mask[..., None] == np.arange(self.num_labels)).astype(np.float32)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        ids = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                ids.append(row[0])

        s1images = [os.path.join(data_dir, "images/s1", data[: data.rfind('lc')]+'s1'+data[data.rfind('lc') :][2 :]) for data in ids]
        s2images = [os.path.join(data_dir, "images/s2", data[: data.rfind('lc')]+'s2'+data[data.rfind('lc') :][2 :]) for data in ids]
        masks = [os.path.join(data_dir, "labels", data) for data in ids]

        valid_s1, valid_s2, valid_masks = [], [], []

        # Multithreaded validation bypasses sequential disk read limits
        print("Validating dataset paths. This may take a moment...")
        
        def check_paths(idx):
            s1, s2, m = s1images[idx], s2images[idx], masks[idx]
            
            # Conditionally check files based on what is actually needed
            valid = True
            if self.imagery in ['s1', 'all']: valid = valid and os.path.exists(s1)
            if self.imagery in ['s2', 'all']: valid = valid and os.path.exists(s2)
            valid = valid and os.path.exists(m)
            
            return (s1, s2, m) if valid else None

        # Execute check using threads
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = executor.map(check_paths, range(len(ids)), chunksize=5000)
            
            for res in results:
                if res:
                    valid_s1.append(res[0])
                    valid_s2.append(res[1])
                    valid_masks.append(res[2])

        self.s1images = valid_s1
        self.s2images = valid_s2
        self.masks = valid_masks

        return self.s1images, self.s2images, self.masks

    def __len__(self):
        return len(self.masks)
    
    def get_labels(self):
        return self.labels
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    def show_image(self, index, show_title=False):
        if self.selection != 'rgb':
            print("The selection parameter must be set to rgb for image visualization")
            
        img, mask = self[index]
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
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.7, 0.5), ncol=2, mode='expand',
                   loc='lower left', prop={'size': 12})

        if self.imagery == 's1' or self.imagery =='s2':
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")

        elif self.imagery == 'all':
            # s1 subplot
            plt.subplot(1, 3, 1)
            plt.imshow(img[:,:,:3])
            plt.title("S1")
            plt.axis("off")

            # s2 subplot
            plt.subplot(1, 3, 2)
            plt.imshow(img[:,:,3:6])
            plt.title("S2")
            plt.axis("off")

            # mask subplot
            plt.subplot(1, 3, 3)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig