import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.patches import Patch
from PIL import Image

from ..base import BaseDataset
from .schemas import OmbriaSchema
from ..utils import image_loader


"""
It is a Sentinel-1 and Sentinel-2 imagery dataset constructed for benchmarking the OmbriaNet deep learning CNN architecture. 
OmbriaNet was designed for adressing the flood mapping problem.
"""

class OMBRIADataset(BaseDataset):

    url = "https://github.com/geodrak/OMBRIA"
    name = "OMBRIA"
    schema = OmbriaSchema
    labels = ["no change","flood"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        if self.csv_file == None: # original train/val/test splits
            img_id = self.data[index]
            split = self.mode

        elif self.csv_file: # new 60/20/20 splits
            img_id = self.data[index][0]
            split = self.data[index][1] # original split of that sample

        # s1 images have a shape of (256 x 256 x 1)
        if self.imagery == 'S1':
            imagery = self.imagery
            pre_image = image_loader(os.path.join(self.data_dir, f'Ombria{imagery}', split, 'BEFORE', f'{imagery}_before_{img_id}.png'))
            if self.selection == "rgb": # (256 x 256 x 3)
                pre_image = np.asarray(Image.fromarray(pre_image).convert('RGB'))
            elif self.selection == "all":
                pre_image = pre_image # original 1 band image
            #normalize image for better visualization
            pre_image = (pre_image - pre_image.min()) / (pre_image.max() - pre_image.min() + 1e-8)

            post_image = image_loader(os.path.join(self.data_dir, f'Ombria{imagery}', split, 'AFTER', f'{imagery}_after_{img_id}.png'))
            if self.selection == "rgb": # (256 x 256 x 3)
                post_image = np.asarray(Image.fromarray(post_image).convert('RGB'))
            elif self.selection == "all":
                pre_image = pre_image # original 1 band image
            #normalize image
            post_image = (post_image - post_image.min()) / (post_image.max() - post_image.min() + 1e-8)

            mask = image_loader(os.path.join(self.data_dir, f'Ombria{imagery}', split, 'MASK', f'{imagery}_mask_{img_id}.png'))

        # s2 images have a shape of (256 x 256 x 3)
        if self.imagery == 'S2':
            imagery = self.imagery
            pre_image = image_loader(os.path.join(self.data_dir, f'Ombria{imagery}', split, 'BEFORE', f'{imagery}_before_{img_id}.png'))
            #normalize image for better visualization
            img_min, img_max = pre_image.min(), pre_image.max()
            pre_image = (pre_image - img_min) / (img_max - img_min + 1e-8)

            post_image = image_loader(os.path.join(self.data_dir, f'Ombria{imagery}', split, 'AFTER', f'{imagery}_after_{img_id}.png'))
            #normalize image
            img_min, img_max = post_image.min(), post_image.max()
            post_image = (post_image - img_min) / (img_max - img_min + 1e-8)

            mask = image_loader(os.path.join(self.data_dir, f'Ombria{imagery}', split, 'MASK', f'{imagery}_mask_{img_id}.png'))

        elif self.imagery == 'all':
            s1_pre_image = image_loader(os.path.join(self.data_dir, f'OmbriaS1', split, 'BEFORE', f'S1_before_{img_id}.png'))
            s1_post_image = image_loader(os.path.join(self.data_dir, f'OmbriaS1', split, 'AFTER', f'S1_after_{img_id}.png'))
            if self.selection == "rgb": # (256 x 256 x 3)
                s1_pre_image = np.asarray(Image.fromarray(s1_pre_image).convert('RGB'))
                s1_post_image = np.asarray(Image.fromarray(s1_post_image).convert('RGB'))

            #normalize image
            s1_pre_image = (s1_pre_image - s1_pre_image.min()) / (s1_pre_image.max() - s1_pre_image.min() + 1e-8)
            s1_post_image = (s1_post_image - s1_post_image.min()) / (s1_post_image.max() - s1_post_image.min() + 1e-8)

            s2_pre_image = image_loader(os.path.join(self.data_dir, f'OmbriaS2', split, 'BEFORE', f'S2_before_{img_id}.png'))
            s2_post_image = image_loader(os.path.join(self.data_dir, f'OmbriaS2', split, 'AFTER', f'S2_after_{img_id}.png'))

            #normalize image
            s2_pre_image = (s2_pre_image - s2_pre_image.min()) / (s2_pre_image.max() - s2_pre_image.min() + 1e-8)
            s2_post_image = (s2_post_image - s2_post_image.min()) / (s2_post_image.max() - s2_post_image.min() + 1e-8)

            pre_image = np.dstack((s1_pre_image, s2_pre_image))
            post_image = np.dstack((s1_post_image, s2_post_image))

            mask = image_loader(os.path.join(self.data_dir, f'OmbriaS2', split, 'MASK', f'S2_mask_{img_id}.png'))
        
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
        if self.csv_file == None: # original train/val/test splits
            split = self.mode
            selection_dir = (os.path.join(data_dir, f'OmbriaS2', split))
            for id in os.listdir(os.path.join(selection_dir, 'AFTER')):
                self.data.append(id[9 : -4])
        
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

        # Create mask visualization
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

        # Add legend at the top
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.05, 0.8, 0.05), ncol=3,
                    loc='center', prop={'size': 12})
        
        if self.imagery == 'all':
            plt.subplot(1, 5, 1)
            plt.imshow(pre_img[:,:,0])
            plt.title("S1 Pre-event")
            plt.axis("off")

            plt.subplot(1, 5, 2)
            plt.imshow(post_img[:,:,0])
            plt.title("S1 Post-event")
            plt.axis("off")

            plt.subplot(1, 5, 3)
            plt.imshow(pre_img[:,:,1:4])
            plt.title("S2 Pre-event")
            plt.axis("off")

            plt.subplot(1, 5, 4)
            plt.imshow(post_img[:,:,1:4])
            plt.title("S2 Post-event")
            plt.axis("off")

            plt.subplot(1, 5, 5)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")
        
        else:
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