from .multiclass_classification import MultiClassClassificationDataset
from ..utils import image_loader

import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

'''
A dataset and deep Learning benchmark for event recognition in aerial videos. The dataset consists of 2864 videos
each with a label from 25 different classes corresponding to an event unfolding 5 seconds.
'''

LABELS = ["Baseball","Basketball","Boating","CarRacing","Concert","Conflict","Constructing","Cycling",
          "Fire","Flood","Harvesting","Landslide","Mudslide","NonEvent","ParadeProtest","Party","Ploughing",
          "PoliceChase","PostEarthquake","ReligiousActivity","Running","Soccer","Swimming","TrafficCollision",
          "TrafficCongestion"]


class ERADataset(MultiClassClassificationDataset):

    url = "https://drive.google.com/file/d/1yxXjDNAq5RAufSgSOE4QmdsNfAc4dOfM/view"
    labels = LABELS
    name = "ERA dataset"

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
        # the data is originally split into about 50/50 train and test splits 
        # csv_files created by BVLabs give a 60/20/20 train/val/test split stratified based on class values
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
            for label in os.listdir(self.data_dir):
                for file_name in os.listdir(os.path.join(self.data_dir,label)):
                    item = (
                        os.path.join(self.data_dir, label, file_name),
                        self.labels.index(label),
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