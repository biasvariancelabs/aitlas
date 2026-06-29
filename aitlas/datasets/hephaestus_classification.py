from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader
import csv
import os
import numpy as np
import pandas as pd

'''
Hephaestus is a large scale multitask dataset created for InSAR (Interferometric SAR) development. 
It revolves around the 44 most active volcanoes globally and is focused around global volcano monitoring. 
This description focuses on the image classification task.
'''

LABELS = ["Sill","Mogi","Dyke","Earthquake","Spheriod","Unidentified","no-data"]


class HephaestusImageClassificationDataset(MultiClassClassificationDataset):

    url = "https://www.dropbox.com/scl/fi/qx99z7tlx6994r7zn4fgh/Hephaestus_Classification.zip?rlkey=lnnmakeu50au0r50yjbjonivx&e=1&dl=0"
    labels = LABELS
    name = "Hephaestus Image Classification dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset()

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

    def load_dataset(self):
        data = []
        if self.csv_file:
            with open(self.csv_file, "r") as f:
                csv_reader = csv.reader(f)
                raw_data = list(csv_reader)

                for index, row in enumerate(raw_data):
                    file_name = row[0]
                    item = (
                        os.path.join(self.data_dir, file_name),
                        int(row[1]),
                    )
                    data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data
    
    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            df["Label"][i] = self.labels[int(df["Label"][i])]
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name", "Label"])
        for i in np.arange(len(df)):
            df["Label"][i] = self.labels[int(df["Label"][i])]
        label_count = df.groupby("Label").count().reset_index()
        label_count.columns = ["Label", "Count"]
        return label_count