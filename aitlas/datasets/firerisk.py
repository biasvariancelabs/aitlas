from .multiclass_classification import MultiClassClassificationDataset
from ..base import BaseDataset
from .schemas import ClassificationDatasetSchema
from ..utils import image_loader

import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np

'''
FireRisk is a remote sensing dataset for fire risk assessment. It contains 7 fire risk classes 
with a total of 91872 labelled images.
'''

LABELS = ["High","Low","Moderate","Non-burnable","Very_High","Very_Low","Water"]

class FireRiskDataset(BaseDataset):

    url = "https://drive.google.com/file/d/1J5GrJJPLWkpuptfY_kgqkiDtcSNP88OP/view"
    labels = LABELS
    name = "FireRisk dataset"
    schema = ClassificationDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

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

    def load_dataset(self, data_dir):
        data = []
        # the dataset is originally split into 80/20 train/test split
        # csv_files created by BVLabs provide a 60/20/20 train/val/test split with a stratification based on class
        if self.csv_file:
            with open(self.csv_file, "r") as f:
                    reader = csv.reader(f)
                    next(reader, None) # skip header
                    for row in reader:
                        file_name = os.path.join(row[1], row[0])
                        label = row[2] 
                        item = (
                            os.path.join(self.data_dir, file_name),
                            self.labels.index(label),            
                        )
                        data.append(item)

        else: # original split
            for labels in os.listdir(data_dir):
                for j in os.listdir(data_dir + labels):
                    file_name = labels + '/' + j
                    item = (
                        os.path.join(self.data_dir, file_name),
                        self.labels.index(labels),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        df = pd.DataFrame(columns=['File name','Label'])
        for i in np.arange(len(self.data)):
            df.loc[i] = [self.data[i][0]] + [self.labels[self.data[i][1]]]
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


    def __len__(self):
        return len(self.data)

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

    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )
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