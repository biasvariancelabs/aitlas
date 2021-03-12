import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import compress
from ..base import BaseDataset
from ..utils import image_loader
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
        return fig

    def show_samples(self):
        df = pd.read_csv(self.dir_path + "/multilabels.txt", sep="\t")
        return df.head(20)

    def show_image(self, index):
        labels_list = list(compress(self.labels, self[index][1]))
        fig = plt.figure(figsize=(8, 6))
        plt.title(f"Image with index {index} from the dataset {self.get_name()}, with labels: {labels_list}")
        plt.axis('off')
        plt.imshow(self[index][0])
        return fig

    def load_dataset(self, dir_path):
        # read labels
        multi_hot_labels = {}
        with open(dir_path + "/multilabels.txt", "rb") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.decode("utf-8")
                labels_list = line[line.find("\t") + 1:].split("\t")
                multi_hot_labels[line[: line.find("\t")]] = np.asarray(
                    list((map(float, labels_list)))
                )

        images = []
        dir = os.path.expanduser(dir_path + "/images")
        # this ensures the image always have the same index numbers
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                multi_hot_label = multi_hot_labels[fname[: fname.find(".")]]
                item = (path, multi_hot_label)
                images.append(item)

        return images
