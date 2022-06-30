import logging
import random
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..base import BaseDataset
from .schemas import So2SatDatasetSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


LABELS = [
    "Compact high_rise",
    "Compact middle_rise",
    "Compact low_rise",
    "Open high_rise",
    "Open middle_rise",
    "Open low_rise",
    "Lightweight low_rise",
    "Large low_rise",
    "Sparsely built",
    "Heavy industry",
    "Dense trees",
    "Scattered trees",
    "Bush or scrub",
    "Low plants",
    "Bare rock or paved",
    "Bare soil or sand",
    "Water",
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

        # we are using sentinel 2 data only for now
        img = self.data["sen2"][index][:, :, [2, 1, 0]].astype(np.float32)
        # Calibration for the optical RGB channels of Sentinel-2 in this dataset.
        img = np.clip(img * 3.5 * 255.0, 0, 255).astype(np.uint8)

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
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def show_samples(self):
        return self.data["label"][:20]

    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.data["sen2"])), size)
        figure, ax = plt.subplots(int(size / 5), 5, figsize=(13.75, 2.8*int(size/5)))
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])  # just show the RGB channel
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
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
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig
