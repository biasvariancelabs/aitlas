import pandas as pd
import numpy as np
import os

from ..utils import image_loader
from .multilabel_classification import MultiLabelClassificationDataset

'''
MultiScene is a large-scale dataset for multi-scene recognition in single aerial images. It contains 
100000 high-resolution images and 36 scene categories.
'''

LABELS = ["apron","baseball field","basktball field","beach","bridge","cemetery","commercial","farmland",
          "woodland","golf course","greenhouse","helipad","lake/pond","oil field","orchard","parking lot",
          "park","pier","port","quarry","railway","residential","river","roundabout","runway","soccer field",
          "solar farm","sparse shrub","stadium","storage tanks","tennis court","train station","wastewater",
          "plant","wind turbine","works","sea"]


class MultiSceneDataset(MultiLabelClassificationDataset):
    url = "https://drive.google.com/drive/folders/17ty4I7HdihLcqRK3k_zDVtQqmPIYwLh2"

    labels = LABELS
    name = "MultiScene"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)

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

        img = image_loader(self.data[index][0])
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(target)
            
        return img, target

    def load_dataset(self, data_dir, csv_file):
        # If not provided initialize the labels from the csv file
        if not self.labels:
            with open(csv_file, "rb") as f:
                header = f.readline().decode("utf-8")
                self.labels = header[header.find(",") + 1:].strip().split(",")
             
        # read labels
        multi_hot_labels = {}
        with open(csv_file, "rb") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                labels_list = line[line.find(",") + 1 :].split(",")
                cleaned_labels = [label.strip() if label.strip() else '0' for label in labels_list]
                multi_hot_labels[line[: line.find(",")]] = np.asarray(
                    list((map(float, cleaned_labels)))
                )

        data = []
        images_folder = os.path.expanduser(data_dir)
        # this ensures the images always have the same index numbers
        for root, _, fnames in sorted(os.walk(images_folder)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if fname[: fname.find(".")] in multi_hot_labels:
                    multi_hot_label = multi_hot_labels[fname[: fname.find(".")]]
                    item = (path, multi_hot_label)
                    data.append(item)

        return data
    
    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",")
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",")
        label_count = pd.DataFrame(df.sum(axis=0)).reset_index()
        label_count.columns = ["Label", "Count"]
        label_count.drop(label_count.index[0], inplace=True)
        return label_count