import csv
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Patch
from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema

"""
Generic dataset for the task of semantic segmentation
"""


class SemanticSegmentationDataset(BaseDataset):
    schema = SegmentationDatasetSchema

    labels = None
    color_mapping = None
    name = None

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.data_dir, self.config.csv_file)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])[:, :, 1] / 255
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)

    def __len__(self):
        return len(self.images)

    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        with open(csv_file, "r") as f:
            csv_reader = csv.reader(f)
            for index, row in enumerate(csv_reader):
                self.images.append(os.path.join(data_dir, row[0] + ".jpg"))
                self.masks.append(os.path.join(data_dir, row[0] + "_m.png"))

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
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

    def show_image(self, index, show_title=False):
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
        # if show_title:
        #    fig.suptitle(
        #        f"Image and mask with index {index} from the {self.get_name()} dataset\n",
        #        fontsize=16,
        #        y=0.82,
        #    )
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
