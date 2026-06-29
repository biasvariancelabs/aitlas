import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import csv

from .change_detection import ChangeDetectionDataset
from .schemas import FAIREOSchema
from ..utils import image_loader


"""
LEVIR-CD+ is an extended version of the original LEVIR-CD dataset, designed for building 
change detection tasks in urban areas using remote sensing imagery.
"""

class LEVIRCDPlusDataset(ChangeDetectionDataset):

    url = "https://drive.google.com/drive/folders/1lJ5GteQy5Z2LVayrKYZOn7_QPImlYP8C"
    name = "LEVIR-CD+"
    schema = FAIREOSchema
    labels = ["no change","building change"]
    color_mapping = [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        if self.csv_file == None: # original train/test splits
            img_id = self.data[index]
            pre_image = image_loader(os.path.join(self.data_dir, 'A', f'{img_id}.png'))
            post_image = image_loader(os.path.join(self.data_dir, 'B', f'{img_id}.png'))
            mask = image_loader(os.path.join(self.data_dir, 'label', f'{img_id}.png'))

        elif self.csv_file: # new 60/20/20 splits
            img_id = self.data[index][0]
            original_split = self.data[index][1]
            pre_image = image_loader(os.path.join(self.data_dir, original_split, 'A', f'{img_id}.png'))
            post_image = image_loader(os.path.join(self.data_dir, original_split, 'B', f'{img_id}.png'))
            mask = image_loader(os.path.join(self.data_dir, original_split, 'label', f'{img_id}.png'))

        # normalize images
        pre_image = (pre_image - pre_image.min()) / (pre_image.max() - pre_image.min() + 1e-8)
        post_image = (post_image - post_image.min()) / (post_image.max() - post_image.min() + 1e-8)

        # translate the mask values from color code to integer
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        for i in np.arange(mask.shape[0]):
            for j in np.arange(mask.shape[1]):
                if mask[i,j] == 0:
                    single_band_mask[i,j] = 0
                elif mask[i,j] == 255:
                    single_band_mask[i,j] = 1
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        self.data = []
        if self.csv_file == None: # original train/test splits
            for id in os.listdir(os.path.join(data_dir, 'A')):
                self.data.append(id[: -4])

        elif self.csv_file: # new 60/20/20 train/val/test splits provided by BVLabs
            with open(self.csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    img_id = row[0] 
                    original_split = row[1]
                    self.data.append((img_id, original_split))

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        data = self[index]
        (pre_img, post_img), mask = data

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

        fig = plt.figure(figsize=(15, 6))
        plt.subplots_adjust(top=0.85, bottom=0.1)

        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
                    loc='center', prop={'size': 12})

        plt.subplot(1, 3, 1)
        plt.imshow(pre_img)
        plt.title("Pre-event")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(post_img)
        plt.title("Post-event")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(img_mask)
        plt.title("Mask")
        plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig