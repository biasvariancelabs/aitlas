import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import compress
from ..base import BaseDataset
from ..utils import image_loader
from .schemas import MultiClassClassificationDatasetSchema

"""
The format of the multiclass classification dataset is:
image_path1,label1
image_path2,label2
...
"""


class MultiClassClassificationDataset(BaseDataset):
    schema = MultiClassClassificationDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema
        BaseDataset.__init__(self, config)

        # load the data
        self.data = self.load_dataset(self.config.csv_file_path)

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
        df = pd.read_csv(self.config.csv_file_path, sep=",", names=["Image path", "Label"])
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ['Label', 'Count']
        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        return fig

    def show_samples(self):
        df = pd.read_csv(self.config.csv_file_path, sep=",", names=["Image path", "Label"])
        return df.head(20)

    def show_image(self, index):
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
                  fontsize=14)
        plt.axis('off')
        plt.imshow(self[index][0])
        return fig

    def load_dataset(self, file_path):
        if not self.labels:
            raise ValueError(
                "You need to provide the list of labels for the dataset"
            )
        data = []
        if file_path:
            with open(file_path, "r") as f:
                csv_reader = csv.reader(f)
                for index, row in enumerate(csv_reader):
                    path = row[0]
                    item = (path, self.labels.index(row[1]))
                    data.append(item)
        return data
