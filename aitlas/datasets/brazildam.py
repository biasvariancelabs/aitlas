from ..base import BaseDataset
from ..utils import image_loader
from .schemas import BrazilDAMSchema

import pandas as pd
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

'''
BrazilDAM is a multi sensor and multitemporal dataset that consists of multispectral images 
of ore tailings dams throughout Brazil. Images are from Sentinel-2 and Landsat 8 satellites.
'''

LABELS = ["nao_barragem","barragem"]


class BrazilDAMDataset(BaseDataset):

    url = "https://drive.google.com/drive/folders/1v1F4faAD8zCm_vocGxILiIUncaRz1pZB"
    labels = LABELS
    name = "BrazilDAM dataset"
    schema = BrazilDAMSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.imagery = self.config.imagery
        self.selection = self.config.selection
        self.bands_s2 = self.config.bands_s2
        self.bands_l8 = self.config.bands_l8
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """

        # consider only Sentinel-2 
        if self.imagery == 'sentinel':
            img = image_loader(self.data[index][0][0])
            satellite = self.data[index][1]
            if self.selection == "rgb":
                img = img[:,:,1:4]
            elif self.selection == "all":
                img = img
            elif self.selection == "bands": 
                if self.bands_s2 == None:
                    print("The config must contain a bands_s2 field with a list of chosen bands")
                idx = []
                s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
                for b in self.bands_s2:
                    if b not in s2_bands:
                        print("The bands must be valid Sentinel-2 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12")
                        break
                    else:
                        idx.append(s2_bands.index(b))
                img = img[:,:,idx]
            # normalize the image
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)


        # consider only Landsat 8
        elif self.imagery == 'landsat':
            img = image_loader(self.data[index][0][0])
            satellite = self.data[index][1]
            if self.selection == "rgb":
                img = img[:,:,1:4]
            elif self.selection == "all":
                img = img
            elif self.selection == "bands": 
                if self.bands_l8 == None:
                    print("The config must contain a bands_l8 field with a list of chosen bands")
                idx = []
                l8_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11"]
                for b in self.bands_l8:
                    if b not in l8_bands:
                        print("The bands must be valid Landsat 8 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11")
                        break
                    else:
                        idx.append(l8_bands.index(b))
                img = img[:,:,idx]
            # normalize the image
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)


        # consider unpaired Sentinel-2 and Landsat 8
        elif self.imagery == 'sentinel and landsat unpaired':
            img = image_loader(self.data[index][0][0])
            satellite = self.data[index][1]
            if satellite == "sentinel": 
                if self.selection == "rgb":
                    img = img[:,:,1:4]
                elif self.selection == "all":
                    img = img
                elif self.selection == "bands": 
                    if self.bands_s2 == None:
                        print("The config must contain a bands_s2 field with a list of chosen bands")
                    idx = []
                    s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
                    for b in self.bands_s2:
                        if b not in s2_bands:
                            print("The bands must be valid Sentinel-2 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12")
                            break
                        else:
                            idx.append(s2_bands.index(b))
                    img = img[:,:,idx]

            elif satellite == "landsat": 
                if self.selection == "rgb":
                    img = img[:,:,1:4]
                elif self.selection == "all":
                    img = img
                elif self.selection == "bands": 
                    if self.bands_l8 == None:
                        print("The config must contain a bands_l8 field with a list of chosen bands")
                    idx = []
                    l8_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11"]
                    for b in self.bands_l8:
                        if b not in l8_bands:
                            print("The bands must be valid Landsat 8 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11")
                            break
                        else:
                            idx.append(l8_bands.index(b))
                    img = img[:,:,idx]

            # normalize the image
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-8)


        # consider S2 and L8 pairs where available
        if self.imagery == 'sentinel and landsat paired':
            s2_img = image_loader(os.path.join(self.data_dir, "sentinel", self.data[index][0][0]))
            if self.selection == "rgb":
                s2_img = s2_img[:,:,1:4]
            elif self.selection == "all":
                s2_img = s2_img
            elif self.selection == "bands":
                if self.bands_s2 == None:
                    print("The config must contain a bands_s2 field with a list of chosen bands")
                idx = []
                s2_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
                for b in self.bands_s2:
                    if b not in s2_bands:
                        print("The bands must be valid Sentinel-2 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12")
                        break
                    else:
                        idx.append(s2_bands.index(b))
                s2_img = s2_img[:,:,idx]
            s2_img = (s2_img - s2_img.min()) / (s2_img.max() - s2_img.min() + 1e-8) # normalize image

            l8_img = image_loader(os.path.join(self.data_dir, "landsat", self.data[index][0][0]))
            if self.selection == "rgb":
                l8_img = l8_img[:,:,1:4]
            elif self.selection == "all":
                l8_img = l8_img
            elif self.selection == "bands": 
                if self.bands_l8 == None:
                    print("The config must contain a bands_l8 field with a list of chosen bands")
                idx = []
                l8_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11"]
                for b in self.bands_l8:
                    if b not in l8_bands:
                        print("The bands must be valid Landsat 8 bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11")
                        break
                    else:
                        idx.append(l8_bands.index(b))
                l8_img = l8_img[:,:,idx]
            l8_img = (l8_img - l8_img.min()) / (l8_img.max() - l8_img.min() + 1e-8)

            img = np.dstack((s2_img, l8_img))
        
        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][0][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target
    
    def load_dataset(self):
        data = []
        if self.csv_file: # csv file created by BVLabs provide a 60/20/20 split with a stratification based on satellite, class and year
            with open(self.csv_file, "r") as f:
                csv_reader = csv.reader(f)
                raw_data = list(csv_reader)

                for row in raw_data[1:]:
                    file_name = row[0]
                    satellite = row[1]
                    label = row[2]
                    year = row[3]

                    item = (
                        os.path.join(self.data_dir, satellite, year, file_name),
                        self.labels.index(label),
                    )

                    if self.imagery == 'sentinel':
                        if satellite == "sentinel":
                            data.append((item, satellite))
                    elif self.imagery == 'landsat':
                        if satellite == 'landsat':
                            data.append((item, satellite))
                    elif self.imagery == 'sentinel and landsat unpaired':
                        data.append((item, satellite))
                    elif self.imagery == 'sentinel and landsat paired':
                        if satellite == "sentinel": # both S2 and L8 have the same file_name if belonging to the same pair
                            if os.path.exists(os.path.join(self.data_dir, "landsat", year, file_name)): # not all S2 have a corresponding L8
                                item = (
                                    os.path.join(year, file_name),
                                    self.labels.index(label),
                                )
                                data.append((item, satellite))

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels
    
    def show_samples(self):
        file_names = [os.path.basename(item[0][0]) for item in self.data]
        labels = [self.labels[item[0][1]] for item in self.data]
        df = pd.DataFrame({'File_name': file_names, 'Label': labels})
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.DataFrame([item for item, _ in self.data], columns=['File_name', 'Label'])
        label_count = df.groupby("Label").count().reset_index()
        label_count['Label'] = ["nao_barragem","barragem"]
        label_count.columns = ["Label", "Count"]
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig

    def show_image(self, index):
        if self.selection != 'rgb':
            print('The selection parameter must be set to rgb for image visualization')

        if self.imagery == 'sentinel' or self.imagery == 'landsat' or self.imagery == 'sentinel and landsat unpaired':
            label = self.labels[self[index][1]]
            fig = plt.figure(figsize=(8, 6))
            plt.title(
                f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
                fontsize=14,
            )
            plt.axis("off")
            plt.imshow(self[index][0])

        elif self.imagery == 'sentinel and landsat paired':    
            s2_img = self[index][0][:,:,:3]
            l8_img = self[index][0][:,:,3:]
            label = self.labels[self[index][1]]

            fig, axs = plt.subplots(1, 2, figsize=(16, 6))

            axs[0].imshow(s2_img)
            axs[0].set_title("S2 image", fontsize=16)
            axs[0].axis("off")

            axs[1].imshow(l8_img)
            axs[1].set_title(f"L8 image", fontsize=16)
            axs[1].axis("off")

            fig.suptitle(
                f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
                fontsize=20,
                y=1.05
            )
            plt.tight_layout()

        return fig
    
    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure