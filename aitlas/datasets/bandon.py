import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import csv

from .semantic_segmentation import BaseDataset
from .schemas import BandonSchema
from ..utils import image_loader


"""
The building change detection with off-nadir aerial images dataset (BANDON) was built to address the problem of 
accurate detection and localization of urban building changes from oblique drone images. The BANDON dataset contains 
2283 pairs of images, 2283 change labels, 1891 BT-flows labels, 1891 pairs of segmentation labels, and 1891 pairs of 
ST-offsets labels (test sets do not provide auxiliary annotations).
"""

class BandonDataset(BaseDataset):

    url = "https://github.com/fitzpchao/BANDON"
    name = "Bandon"
    schema = BandonSchema
    labels = ["no change","change"]
    color_mapping = [[0,0,0],[255,255,255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.split = self.config.split
        self.pre_image, self.post_image, self.change_image = self.load_dataset(self.data_dir)

    def __getitem__(self, index):

        pre_image = image_loader(self.data_dir + self.pre_image[index])
        post_image = image_loader(self.data_dir + self.post_image[index])
        # normalize images
        pre_image = (pre_image - pre_image.min()) / (pre_image.max() - pre_image.min() + 1e-8)
        post_image = (post_image - post_image.min()) / (post_image.max() - post_image.min() + 1e-8)

        mask = image_loader(self.data_dir + self.change_image[index])  
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return (pre_image, post_image), mask


    def load_dataset(self, data_dir):
        # The dataset contains change labels between 3 time steps: t1 vs t2, t2 vs t3 and t1 vs t3. 
        # Which of these 3 is available depends on the sample.
        # The information (file names and corresponding mask) is found in the txt files (and csv files).

        pre_image = []
        post_image = []
        change_image = []

        if self.split == 'original':
            with open(self.csv_file, 'r') as f:
                for line in f.readlines():
                    img_path = line.split(' ')
                    pre_image.append(img_path[0])
                    post_image.append(img_path[1])
                    change_image.append(img_path[2])

        elif self.split == 'new': # 60/20/20 split created by BVLabs
            with open(self.csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    img_path = row[1].split(' ')
                    original_split = row[0]
                    pre_image.append(original_split + img_path[0])
                    post_image.append(original_split + img_path[1])
                    change_image.append(original_split + img_path[2])

        return pre_image, post_image, change_image

    def __len__(self):
        return len(self.pre_image)
    
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

        # Create subplot with space for legend on top
        plt.subplots_adjust(top=0.85, bottom=0.1)

        # Add legend at the top
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
        for i in np.arange(len(self.pre_image)):
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