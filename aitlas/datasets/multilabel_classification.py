import os
import random
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..base import BaseDataset
from ..utils import image_loader, load_voc_format_dataset
from .schemas import MultiLabelClassificationDatasetSchema


"""
The MultiLabelClassificationdataset is using the Pascal VOC data format
"""


class MultiLabelClassificationDataset(BaseDataset):
    schema = MultiLabelClassificationDatasetSchema

    def __init__(self, config):
        # now call the constuctor to validate the schema
        BaseDataset.__init__(self, config)

        # this can be overridden if needed
        self.image_loader = image_loader

        # load the data
        self.dir_path = self.config.root
        self.data = self.load_dataset(self.dir_path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = self.image_loader(self.data[index][0])
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(self.data[index][1])
        return img, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        df = pd.read_csv(self.dir_path + "/multilabels.txt", sep="\t")
        label_count = pd.DataFrame(df.sum(axis=0)).reset_index()
        label_count.columns = ["Label", "Count"]
        label_count.drop(label_count.index[0], inplace=True)
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title("Image distribution for {}".format(self.get_name()), fontsize=18)
        return fig

    def show_samples(self):
        df = pd.read_csv(self.dir_path + "/multilabels.txt", sep="\t")
        return df.head(20)

    def show_image(self, index):
        labels_list = list(compress(self.labels, self[index][1]))
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with labels:\n "
            f"{str(labels_list).strip('[]')}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def show_batch(self, size):
        if size % 3:
            raise ValueError("The provided size should be divided by 4!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure_height = int(size / 3) * 4
        figure, ax = plt.subplots(int(size / 3), 3, figsize=(20, figure_height))
        figure.suptitle(
            "Example images with labels from {}".format(self.get_name()), fontsize=32
        )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])
            labels_list = list(compress(self.labels, self[image_index][1]))
            str_label_list = ""
            if len(labels_list) > 4:
                str_label_list = f"{str(labels_list[0:4]).strip('[]')}\n"
                str_label_list += f"{str(labels_list[4:]).strip('[]')}\n"
            else:
                str_label_list = f"{str(labels_list).strip('[]')}\n"
            axes.set_title(str_label_list, fontsize=18)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        # figure.subplots_adjust(top=0.88)
        return figure

    def load_dataset(self, dir_path):
        return load_voc_format_dataset(dir_path)
