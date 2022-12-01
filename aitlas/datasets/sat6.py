import random
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns

from ..base import BaseDataset
from .schemas import MatDatasetSchema


"""
The format of the mat dataset is:
train_x 28x28x4x400000 uint8 (containing 400000 training samples of 28x28 images each with 4 channels)
train_y 400000x6 uint8 (containing 6x1 vectors having labels for the 400000 training samples)
test_x  28x28x4x100000 uint8 (containing 100000 test samples of 28x28 images each with 4 channels)
test_y  100000x6 uint8 (containing 6x1 vectors having labels for the 100000 test samples)
"""

LABELS = ["buildings", "barren land", "trees", "grassland", "roads", "water bodies"]


class SAT6Dataset(BaseDataset):
    schema = MatDatasetSchema

    url = "http://csc.lsu.edu/~saikat/deepsat/"
    labels = LABELS
    name = "SAT-6 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the data
        self.mode = self.config.mode
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset(self.config.mat_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = self.data[index][0]
        # apply transformations
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
        mat_data = scipy.io.loadmat(self.config.mat_file)
        img_labels = mat_data[f"{self.mode}_y"].transpose()

        indices = None
        if self.csv_file:
            with open(self.csv_file) as infile:
                data = csv.reader(infile)
                indices = [int(row[0]) for row in data]
                indices = sorted(indices, reverse=True)
        if indices:
            # generate list of indices
            indices_range = list(range(len(img_labels)))
            for idx in indices:
                if idx < len(img_labels):
                    indices_range.pop(idx)

            img_labels = np.delete(img_labels, indices_range, 0)

        data = list(np.where(img_labels == 1)[1])
        res_list = [[i, self.labels[index]] for i, index in enumerate(data)]
        df = pd.DataFrame(res_list, columns=["id", "Label"])
        label_count = df.groupby("Label").count().reset_index()
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
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(int(size / 5), 5, figsize=(13.75, 2.8*int(size/5)))
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

    def load_dataset(self, mat_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        indices = None
        if self.csv_file:
            with open(self.csv_file) as infile:
                data = csv.reader(infile)
                indices = [int(row[0]) for row in data]
                indices = sorted(indices, reverse=True)

        data = []
        if mat_file:
            mat_data = scipy.io.loadmat(mat_file)
            images = mat_data[f"{self.mode}_x"].transpose(3, 0, 1, 2)
            img_labels = mat_data[f"{self.mode}_y"].transpose()

            if indices:
                # generate list of indices
                indices_range = list(range(len(images)))
                for idx in indices:
                    if idx < len(images):
                        indices_range.pop(idx)

                images = np.delete(images, indices_range, 0)
                img_labels = np.delete(img_labels, indices_range, 0)

            data = list(zip(images[:, :, :, 0:3], np.where(img_labels == 1)[1]))

        return data

    def re_map_labels(self, labels_remapping):
        # re mapp the labels
        tmp_data = []
        if self.data:
            for i, (img, label) in enumerate(self.data):
                if label in labels_remapping.keys():
                    tmp_data.append((img, labels_remapping[label]))
        self.data = tmp_data
