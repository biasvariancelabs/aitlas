import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import csv
import numpy as np
from PIL import Image

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset

'''
The dataset contains 102 images from the Gaofen-3 satellite and 108 images from Sentinel-1. The 
ships in the images were annotated with a bounding box. The data can be used to develop object 
detectors for multi-scale and small object detection.
'''

class SARShipDetectionDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "SAR ship dataset"
    url = "https://drive.google.com/drive/folders/1wcyV2nzcibh3EoaTuJ67HoS6SqakeGB9?usp=drive_link"

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file 

        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg")) 

        # images have 1 band or 3 bands
        # unify to 3 bands images
        if len(image.shape) == 2: # 1 band image with shape = (256, 256)
            image = np.asarray(Image.fromarray(image).convert('RGB')) / 255
        else:
            image = image / 255
            
        img_h, img_w = image.shape[:2]

        # annotation file
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.txt")
        annot = open(annot_file_path, "r")
        annot = annot.readlines()
        boxes = []
        labels = []

        # box coordinates for txt files are extracted
        for annotation in annot:
            lines = annotation[:-1] #removes /n
            elements = lines.split()
            # bounding box
            xmin = round((float(elements[1])-(float(elements[3])/2))*img_w)
            xmax = round((float(elements[1])+(float(elements[3])/2))*img_w)

            ymin = round((float(elements[2])-(float(elements[4])/2))*img_h)
            ymax = round((float(elements[2])+(float(elements[4])/2))*img_h)

            if xmax > xmin and ymax > ymin:
                labels.append(int(1))
                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 
  
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def load_dataset(self, data_dir, annotation_dir, csv_file):
        labels = []
        annotations = []
        data = []

        ids = os.listdir(data_dir)

        if csv_file: # train/validation/test split
            ids = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    image_id = row[0].split('/')[1]
                    ids.append(image_id)
            
        for img in ids:
            img_name = img[: img.rfind('.jpg')]
            annot_file_path = os.path.join(annotation_dir, img_name +'.txt') 
            annot = open(annot_file_path, "r")
            annot = annot.readlines()

            image_file_path = os.path.join(data_dir, f"{img_name}.jpg")
            image = cv2.imread(image_file_path)
            img_h, img_w = image.shape[:2]

            labels = ["no-data","ship"]
            
            # box coordinates for txt files are extracted
            for annotation in annot:

                lines = annotation[:-1] #removes /n
                elements = lines.split()
                # bounding box
                xmin = round((float(elements[1])-(float(elements[3])/2))*img_w)
                xmax = round((float(elements[1])+(float(elements[3])/2))*img_w)

                ymin = round((float(elements[2])-(float(elements[4])/2))*img_h)
                ymax = round((float(elements[2])+(float(elements[4])/2))*img_h)

                if xmax > xmin and ymax > ymin:
                    annotations.append({"label": int(1)})

            name = img[: img.rfind('.jpg')]
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