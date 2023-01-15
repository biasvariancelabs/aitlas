import random
from itertools import compress
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2
import math

from ..base import BaseDataset
from ..utils import image_loader, load_voc_format_dataset
from .schemas import ClassificationDatasetSchema


"""
The MultiLabelClassificationdataset is using the Pascal VOC data format
"""


class MultiLabelClassificationDataset(BaseDataset):
    schema = ClassificationDatasetSchema

    def __init__(self, config):
        # now call the constuctor to validate the schema
        super().__init__(config)

        # this can be overridden if needed
        self.image_loader = image_loader

        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset(self.data_dir, self.csv_file)

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
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep="\t")
        label_count = pd.DataFrame(df.sum(axis=0)).reset_index()
        label_count.columns = ["Label", "Count"]
        label_count.drop(label_count.index[0], inplace=True)
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
        df = pd.read_csv(self.csv_file, sep="\t")
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

    def show_batch(self, size, show_title=True):
        if size % 4:
            raise ValueError("The provided size should be divided by 4!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(int(size / 4), 4, figsize=(13.75, 2.0*int(size/4)))
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            labels_list = list(compress(self.labels, self[image_index][1]))
            height, width, depth = self[image_index][0].shape
            white_image = np.zeros([height, width, 3], dtype=np.uint8)
            white_image.fill(255)
            text = '\n'. join(labels_list)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = width/600 * 2.5
            font_thickness = math.ceil(width/600 * 4)
            x = 30

            for i, line in enumerate(text.split('\n')):
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                gap = textsize[1] + 5
                y = textsize[1] + i * gap
                cv2.putText(white_image, line, (x, y), font,
                            font_size,
                            (0, 0, 0),
                            font_thickness,
                            lineType=cv2.LINE_AA)

            display_image = np.hstack((self[image_index][0], white_image))
            axes.imshow(display_image)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axis('off')
        figure.tight_layout()
        return figure

    def load_dataset(self, data_dir, csv_file):
        # If not provided initialize the labels from the csv file
        if not self.labels:
            with open(csv_file, "rb") as f:
                header = f.readline().decode("utf-8")
                self.labels = header[header.find("\t") + 1:-1].split("\t")
        return load_voc_format_dataset(data_dir, csv_file)

    def labels_stats(self):
        min_number = float("inf")
        max_number = float("-inf")
        average_number = 0
        for img, labels in self.data:
            if sum(labels) < min_number:
                min_number = sum(labels)

            if sum(labels) > max_number:
                max_number = sum(labels)

            average_number += sum(labels)

        return (
            f"Minimum number of labels: {min_number}, Maximum number of labels: {max_number}, "
            f"Average number of labels: {average_number/len(self.data)}"
        )

    def re_map_labels(self, labels_remapping, map_size):
        # re mapp the labels
        tmp_data = []
        if self.data:
            for i, (path, one_hot_encoded_labels) in enumerate(self.data):
                new_one_hot_encoded_labels = np.zeros(map_size, dtype=float)
                for j, label in enumerate(one_hot_encoded_labels):
                    if j in labels_remapping.keys() and label == 1:
                        new_one_hot_encoded_labels[labels_remapping[j]] = 1

                tmp_data.append((path, new_one_hot_encoded_labels))
        self.data = tmp_data
