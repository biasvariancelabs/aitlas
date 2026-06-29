from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader
import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import fnmatch

'''
The Global WiDS team, the West Big Data Innovation Hub, and the WiDS Datathon Committee have partnered 
with Planet and Figure Eight to bring more than 20,000 labeled satellite images to participants. The WiDS 
Datathon challenge is to create a model that predicts the presence of oil palm plantations in satellite imagery.
'''

class WiDSDatathon2019Dataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/competitions/widsdatathon2019"
    labels = ["no plantation","plantation"] 
    name = "WiDS Datathon 2019 dataset"

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
        # the dataset is originally split into about 70/20/10 train/val/test splits
        # csv_files created by BVLabs give a new 60/20/20 train/val/test split
        if self.csv_file:
            if fnmatch.fnmatch(self.csv_file, '*widsdatathon2019_*'): # new split
                with open(self.csv_file, "r") as f:
                    csv_reader = csv.reader(f)
                    raw_data = list(csv_reader)[1 :]
                    for index, row in enumerate(raw_data):
                        file_name = os.path.join(row[3], row[0][: 9] + '.jpg')
                        if row[1] == '0':
                            label = 'no plantation'
                        elif row[1] == 0:
                            label = 'no plantation'
                        else:
                            label = 'plantation'
                        item = (
                            os.path.join(self.data_dir, file_name),
                            self.labels.index(label),
                        )
                        data.append(item)

            else: # original split
                with open(self.csv_file, "r") as f:
                    csv_reader = csv.reader(f)
                    raw_data = list(csv_reader)[1 :]
                    for index, row in enumerate(raw_data):
                        file_name = row[0][: 9] + '.jpg'
                        if file_name not in os.listdir(self.data_dir):
                            None
                        else: 
                            if row[1] == '0':
                                label = 'no plantation'
                            elif row[1] == 0:
                                label = 'no plantation'
                            else:
                                label = 'plantation'
                            item = (
                                os.path.join(self.data_dir, file_name),
                                self.labels.index(label),
                            )
                            data.append(item)

        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        return data

    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",", names=["File name","Label","Score"])
        df = df.drop(["Score"],axis=1)
        for j in np.arange(len(df)):
            if df["Label"][j] == '0':
                df["Label"][j] = 'no plantation'
            else:
                df["Label"][j] = 'plantation'
        df = df.drop([0])
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