from ..base import BaseDataset
from ..utils import image_loader
from .schemas import TallosSchema

import random
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import math
import os
import ast
import csv
import json
import fnmatch

'''
The dataset was created to improve the diversity and spatial coverage of FoMo-Bench. This dataset couples 
manual tree inventories with multispectral and SAR timeseries imagery for a fine grained multi-label 
classification challenge. It is based on the global Tallo database.
'''


# 1364 classes. Genus column of the csv file.
# Sentinel-1 images are not associated to any label and are therefore not considered in this dataloader


class TalloSDataset(BaseDataset):
    url = "https://github.com/RolnickLab/FoMo-Bench/tree/main"
    name = "TalloS"
    schema = TallosSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        self.image_loader = image_loader

        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.annotations = self.config.annotations
        self.task = self.config.task
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.bands_s2 = self.config.bands_s2
        self.labels, self.data = self.load_dataset(self.data_dir, self.annotations, self.csv_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a multi-hot vector.
        """
        # load image
        if self.imagery == "sentinel2":
            if self.selection == "all":
                img = self.image_loader(os.path.join(self.data_dir, self.data[index][0]))
            if self.selection == "rgb":
                img = self.image_loader(os.path.join(self.data_dir, self.data[index][0]))[:,:,1:4]
            elif self.selection == "bands":
                if self.bands_s2 == None:
                    print("The config must contain a bands_s2 field with a list of chosen bands")
                img = self.image_loader(os.path.join(self.data_dir, self.data[index][0]))
                idx = []
                s2_bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"] # s2 bands available for this dataset
                for b in self.bands_s2:
                    if b not in s2_bands:
                        print("The bands must be valid and available Sentinel-2 bands, i.e. one of B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12")
                        break
                    else:
                        idx.append(s2_bands.index(b))
                img = img[:,:,idx]
            # normalize image
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        elif self.imagery == "dem": 
            img = self.image_loader(os.path.join(self.data_dir, self.data[index][0]))

        elif self.imagery == "all":
            # Sentinel-2
            if self.selection == "all":
                s2_img = self.image_loader(os.path.join(self.data_dir, self.data[index][0]))
            if self.selection == "rgb":
                s2_img = self.image_loader(os.path.join(self.data_dir, self.data[index][0]))[:,:,1:4]
            elif self.selection == "bands":
                if self.bands_s2 == None:
                    print("The config must contain a bands_s2 field with a list of chosen bands")
                s2_img = self.image_loader(os.path.join(self.data_dir, self.data[index][0]))
                idx = []
                s2_bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"] # s2 bands available for this dataset
                for b in self.bands_s2:
                    if b not in s2_bands:
                        print("The bands must be valid and available Sentinel-2 bands, i.e. one of B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12")
                        break
                    else:
                        idx.append(s2_bands.index(b))
                s2_img = s2_img[:,:,idx]
            # normalize image
            s2_img = (s2_img - s2_img.min()) / (s2_img.max() - s2_img.min() + 1e-8)

            # DEM
            dem_id = "/".join(self.data[index][0].split('/')[:-1]) + '/DEM.tif'
            dem_img = self.image_loader(os.path.join(self.data_dir, dem_id))

            img = np.dstack((s2_img, dem_img))

        if self.transform:
            img = self.transform(img)
            
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def load_dataset(self, data_dir, annotations, csv_file):
        # if not provided initialize the labels from the csv file
        if not self.labels:
            df = pd.read_csv(annotations)
            # convert string list to actual Python list
            df["genus"] = df["genus"].apply(ast.literal_eval)
            # flatten all species into one list
            all_species = [sp for sublist in df["genus"] for sp in sublist]
            # get unique species
            self.labels = sorted(set(all_species))

            # write the labels list in a csv file (optional)
            main_dir = "/".join(data_dir.split('/')[:-2])
            with open(os.path.join(main_dir,"labels_list.csv"), "w") as f:
                for label in self.labels:
                    f.write(label+"\n")
             
        # original train/val/test splits
        if fnmatch.fnmatch(csv_file, "*.json"):
            self.data = []
            # read labels
            with open(csv_file, "r") as f:
                data = json.load(f)

            samples = data[self.task]
            # discard images with no labels
            samples = [sample for sample in samples if any(v > 0 for v in sample["label"])]

            for sample in samples:
                # full image path
                if self.imagery == "dem":
                    path = "/".join(sample[self.imagery].split('/')[2:])
                elif self.imagery == "sentinel2":
                    path = "/".join(sample[self.imagery].split('/')[1:])
                elif self.imagery == "all":
                    path = "/".join(sample["sentinel2"].split('/')[1:]) # both S2 and DEM images have the same parent path
                
                # multi-hot label
                multihot_label = np.asarray(sample["label"])

                if os.path.exists(os.path.join(data_dir, path)):
                    self.data.append((path, multihot_label))

        # new 60/20/20 splits created by BVLabs
        elif fnmatch.fnmatch(csv_file, "*.csv"):
            new_splits = []
            self.data = []
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for file in reader:
                    new_splits.append(file[0])

            # obtain the full list of images path and associated multihot label before keeping only the training, validation or testing images
            for task in ['train','val','test']:
                json_labels = f"{main_dir}/splits/{task}_samples.json"
                with open(json_labels, "r") as f:
                    data = json.load(f)
                samples = data[task]
                # discard images with no labels
                samples = [sample for sample in samples if any(v > 0 for v in sample["label"])]

                for sample in samples:
                    # full image path
                    if self.imagery == "dem":
                        path = "/".join(sample[self.imagery].split('/')[2:])
                    elif self.imagery == "sentinel2":
                        path = "/".join(sample[self.imagery].split('/')[1:])
                    elif self.imagery == "all":
                        path = "/".join(sample["sentinel2"].split('/')[1:]) # both S2 and DEM images have the same parent path
                    
                    # multi-hot label
                    multihot_label = np.asarray(sample["label"])

                    img_id = "/".join(path.split('/')[:-1])
                    if img_id in new_splits:
                        self.data.append((path, multihot_label))

        return self.labels, self.data

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels


    def show_image(self, index):
        if self.selection != "rgb":
            print("The selection parameter must be set to rgb for visualization")
        labels_list = list(compress(self.labels, self[index][1]))

        if self.imagery == 'all':
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(
            f"Image with index {index} from the dataset {self.get_name()}, with labels:\n"
            f"{', '.join(labels_list)}",
            fontsize=14,
            )

            axes[0].imshow(self[index][0][:,:,:3])
            axes[0].set_title('Sentinel-2', fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(self[index][0][:,:,3:])
            axes[1].set_title('DEM', fontsize=12)
            axes[1].axis("off")
            
        else: 
            fig = plt.figure(figsize=(8, 6))
            plt.title(
                f"Image with index {index} from the dataset {self.get_name()}, with labels:\n "
                f"{str(labels_list).strip('[]')}\n",
                fontsize=14,
            )
            plt.axis("off")
            plt.imshow(self[index][0])
        
        return fig

    def show_batch(self, size, show_title=True):
        if size % 4:
            raise ValueError("The provided size should be divided by 4!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(int(size / 4), 4, figsize=(13.75, 2.0*int(size/4)))
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            labels_list = list(compress(self.labels, self[image_index][1]))
            height, width, depth = self[image_index][0].shape
            white_image = np.zeros([height, width, 3], dtype=np.uint8)
            white_image.fill(255)
            text = '\n'. join(labels_list)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = width/600 * 2.5
            font_thickness = math.ceil(width/600 * 4)
            x = 30

            for i, line in enumerate(text.split('\n')):
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                gap = textsize[1] + 5
                y = textsize[1] + i * gap
                cv2.putText(white_image, line, (x, y), font,
                            font_size,
                            (0, 0, 0),
                            font_thickness,
                            lineType=cv2.LINE_AA)

            display_image = np.hstack((self[image_index][0], white_image))
            axes.imshow(display_image)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axis('off')
        figure.tight_layout()

        return figure
    
    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",")
        return df.head(20)
    
    def data_distribution_table(self):
        with open(self.csv_file, "r") as f:
            data = json.load(f)

        samples = data[self.task]
        labels = np.array([sample["label"] for sample in samples])
        counts = labels.sum(axis=0).astype(int) 

        label_count = pd.DataFrame({
            "Label": self.labels,
            "Count": counts
        })

        return label_count
    
    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        # keep the 50 majority classes for display
        # sort by count (descending)
        label_count = label_count.sort_values(by="Count", ascending=False)
        # keep top 50
        label_count = label_count.head(50)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig