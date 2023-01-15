import csv
import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import ClassificationDatasetSchema


"""
The format of the multiclass classification dataset is:
image_path1,label1
image_path2,label2
...
"""


class MultiClassClassificationDataset(BaseDataset):
    schema = ClassificationDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = image_loader(self.data[index][0])
        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
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

    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        return df.head(20)

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

    def load_dataset(self):
        data = []
        if self.csv_file:
            with open(self.csv_file, "r") as f:
                csv_reader = csv.reader(f)
                raw_data = list(csv_reader)

                # If not provided initialize the labels from the csv file
                if not self.labels:
                    self.labels = []
                    for index, row in enumerate(raw_data):
                        self.labels.append(row[1])
                    self.labels = list(sorted(set(self.labels)))

                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    item = (
                        os.path.join(self.data_dir, file_name),
                        self.labels.index(row[1]),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def re_map_labels(self, labels_remapping):
        # re mapp the labels
        tmp_data = []
        if self.data:
            for i, (path, label) in enumerate(self.data):
                if label in labels_remapping.keys():
                    tmp_data.append((path, labels_remapping[label]))
        self.data = tmp_data
