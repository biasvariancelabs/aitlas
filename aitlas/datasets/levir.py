import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset

class LEVIRDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir

        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.png")) / 255.0
        img_h, img_w = image.shape[:2]

        # annotation file
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.txt")
        annot = open(annot_file_path, "r")
        annot = annot.readlines()
        if np.shape(annot)[0] != 0:
            boxes = []
            labels = []

            # box coordinates for txt files are extracted
            for annotation in annot:
                lines = annotation[1 :-1] #removes /n
                elements = lines.split(' ')
                # bounding box
                xmin = float(elements[1])*img_w - (float(elements[3])*img_w)/2
                ymin = float(elements[2])*img_h - (float(elements[4])*img_h)/2

                xmax = float(elements[1])*img_w + (float(elements[3])*img_w)/2
                ymax = float(elements[2])*img_h + (float(elements[4])*img_h)/2

                if xmax > xmin and ymax > ymin:
                    labels.append(int(1))
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 

        return self.apply_transformations(image, target)

    def load_dataset(self, data_dir, annotation_dir):
        labels = []
        annotations = []
        data = []
        for img in os.listdir(data_dir):
            annot_file_path = os.path.join(annotation_dir, img[: img.rfind('.png')]+'.txt') 
            annot = open(annot_file_path, "r")
            annot = annot.readlines()
            if np.shape(annot)[0] != 0:
                img_name = img[: img.rfind('.png')]
                image_file_path = os.path.join(data_dir, f"{img_name}.png")
                image = cv2.imread(image_file_path)
                img_h, img_w = image.shape[:2]

                labels = ["no-data","ship"]
                
                # box coordinates for txt files are extracted
                
                for annotation in annot:

                    lines = annotation[1 :-1] #removes /n and empty space
                    elements = lines.split(" ")
                    # bounding box
                    xmin = float(elements[1])*img_w - (float(elements[3])*img_w)/2
                    ymin = float(elements[2])*img_h - (float(elements[4])*img_h)/2

                    xmax = float(elements[1])*img_w + (float(elements[3])*img_w)/2
                    ymax = float(elements[2])*img_h + (float(elements[4])*img_h)/2

                    if xmax > xmin and ymax > ymin:
                        annotations.append({"label": int(1)})

                name = img[: img.rfind('.png')]
                data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["ship"], True)
        df_count.columns = ["Label", "Count"]
        
        return df_count

    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig