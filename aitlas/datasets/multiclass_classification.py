import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

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
        BaseDataset.__init__(self, config)

        # load the data
        self.dir_path = self.config.dir_path
        self.csv_file_path = self.config.csv_file_path
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
            target = self.target_transform(self.data[index][1])
        return img, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file_path, sep=",", names=["File name", "Label"])
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ['Label', 'Count']
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title("Image distribution for {}".format(self.get_name()), pad=20, fontsize=18)
        return fig

    def show_samples(self):
        df = pd.read_csv(self.csv_file_path, sep=",", names=["File name", "Label"])
        return df.head(20)

    def show_image(self, index):
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
                  fontsize=14)
        plt.axis('off')
        plt.imshow(self[index][0])
        return fig

    def show_batch(self, size):
        if size % 3:
            raise ValueError(
                "The provided size should be divided by 3!"
            )
        image_indices = random.sample(range(0, len(self.data)), size)
        figure_height = int(size / 3) * 4
        figure, ax = plt.subplots(int(size / 3), 3, figsize=(20, figure_height))
        figure.suptitle("Example images with labels from {}".format(self.get_name()), fontsize=32, y=1.006)
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure

    def load_dataset(self):
        if not self.labels:
            raise ValueError(
                "You need to provide the list of labels for the dataset"
            )
        data = []
        if self.csv_file_path:
            with open(self.csv_file_path, "r") as f:
                csv_reader = csv.reader(f)
                for index, row in enumerate(csv_reader):
                    file_name = row[0]
                    item = (os.path.join(self.dir_path, file_name), self.labels.index(row[1]))
                    data.append(item)
        return data
