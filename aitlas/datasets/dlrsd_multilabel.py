from .multilabel_classification import MultiLabelClassificationDataset
from ..utils import image_loader

import pandas as pd
import numpy as np
import os

'''
DLRSD (Dense Labeling Remote Sensing Dataset) is a dense labeling dataset that can be used for multi-label tasks such as remote sensing image retrieval (RSIR) and 
classification, and other pixel-based tasks like semantic segmentation. It has 21 broad categories with 100 images per class, and 17 segmentation classes. This 
dataloader considers the multi-label classification subset.
'''

LABELS = ["airplane","baresoil","buildings","cars","chaparral","court","dock","field","grass","mobilehome","pavement","sand","sea","ship","tanks","trees","water"]


class DLRSDMultilabelDataset(MultiLabelClassificationDataset):
    url = "https://sites.google.com/view/zhouwx/dataset"

    labels = LABELS
    name = "DLRSD multilabel"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
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
            tuple: (image, target) where target is a multi-hot vector.
        """
        # load image
        img = self.image_loader(self.data[index][0])
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

        images = []
        images_folder = os.path.expanduser(data_dir)
        # this ensures the image always have the same index numbers
        for root, _, fnames in sorted(os.walk(images_folder)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if fname[: fname.find(".")] in multi_hot_labels:
                    multi_hot_label = multi_hot_labels[fname[: fname.find(".")]]
                    item = (path, multi_hot_label)
                    images.append(item)

        return images
    
    def show_samples(self):
        df = pd.read_csv(self.csv_file, sep=",")
        return df.head(20)
    
    def data_distribution_table(self):
        df = pd.read_csv(self.csv_file, sep=",")
        label_count = pd.DataFrame(df.sum(axis=0)).reset_index()
        label_count.columns = ["Label", "Count"]
        label_count.drop(label_count.index[0], inplace=True)
        return label_count