import logging
import os
import random
from itertools import compress

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ..base import BaseDataset
from .schemas import So2SatDatasetSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


LABELS = [
    "1_compact_high_rise",
    "2_compact_middle_rise",
    "3_compact_low_rise",
    "4_open_high_rise",
    "5_open_middle_rise",
    "6_open_low_rise",
    "7_lightweight_low_rise",
    "8_large_low_rise",
    "9_sparsely_built",
    "10_heavy_industry",
    "A_dense_trees",
    "B_scattered_trees",
    "C_bush_scrub",
    "D_low_plants",
    "E_bare_rock_or_paved",
    "F_bare_soil_or_sand",
    "G_water",
]


class So2SatDataset(BaseDataset):
    """
        So2Sat dataset version 2 (contains train, validation and test splits)

        So2Sat LCZ42 is a dataset consisting of corresponding synthetic aperture radar and multispectral optical image
        data acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and a corresponding local climate
        zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions
        of the world, and comes with a split into fully independent, non-overlapping training, validation, and test sets.
    """

    url = "https://dataserv.ub.tum.de/s/m1483140/download?path=%2F&files=testing.h5"
    name = "So2Sat dataset"
    schema = So2SatDatasetSchema
    labels = LABELS

    def __init__(self, config):
        super().__init__(config)

        self.file_path = self.config.h5_file
        self.data = h5py.File(
            self.file_path
        )  # TODO: we should close this file eventually

    def __getitem__(self, index):
        label = self.data["label"][index]

        # we are using sentinel 2 data onyl for now
        img = self.data["sen2"][index][:, :, 2:5]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, np.where(label == 1.0)[0][0]

    def __len__(self):
        return self.data["label"].shape[0]

    def get_labels(self):
        return self.labels

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

    def show_samples(self):
        return self.data["label"][:20]

    def show_batch(self, size):
        if size % 3:
            raise ValueError("The provided size should be divided by 3!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure_height = int(size / 3) * 4
        figure, ax = plt.subplots(int(size / 3), 3, figsize=(20, figure_height))
        figure.suptitle(
            "Example images with labels from {}".format(self.get_name()),
            fontsize=32,
            y=1.006,
        )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])  # just show the RGB channel
            labels_list = list(compress(self.labels, self[image_index][1]))
            str_label_list = ""
            if len(labels_list) > 4:
                str_label_list = f"{str(labels_list[0:4]).strip('[]')}\n"
                str_label_list += f"{str(labels_list[4:]).strip('[]')}\n"
            else:
                str_label_list = f"{str(labels_list).strip('[]')}\n"
            axes.set_title(str_label_list[:-1], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure

    def data_distribution_table(self):
        sums = np.sum(self.data["label"], axis=0)
        label_count = pd.DataFrame(
            list(zip(self.labels, sums)), columns=["Label", "Count"]
        )
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title(
            "Image distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
