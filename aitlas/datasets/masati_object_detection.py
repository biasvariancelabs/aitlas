import os
from xml.etree import ElementTree as et
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv 
from PIL import Image

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionPascalSchema
from .object_detection import BaseObjectDetectionDataset

'''
The MASATI (MAritime SATellite Imagery) dataset provides maritime scenes of optical aerial images. 
It can be used to evaluate ship detection methods. It contains 7389 satellite images and can be used 
for image classification, or object detection for the 'coast_ship', 'ship' and 'multi' classes. This 
description considers the object detection task.
'''

class MASATIObjectDetectionDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionPascalSchema
    name = "MASATI Object Detection"
    url = "https://docs.google.com/forms/d/e/1FAIpQLSe1qeo_XH2v_obix_huLVVqWpGL_1DAZlMTVlEeMuy-EHdH9w/viewform?c=0&w=1"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.image_dir = self.config.image_dir
        self.annotations_dir = self.config.annotations_dir
        self.csv_file = self.config.csv_file 

        self.labels, self.annotations, self.data = self.load_dataset(
            self.image_dir,self.annotations_dir,self.csv_file
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.image_dir, f"{img_name}.png")) / 255.0
        if np.ndim(image) == 2:
            image = np.asarray(Image.fromarray(image).convert('RGB'))
        if np.shape(image)[2] == 4:
            image = image[:,:,0:3]

        # annotation file
        label_id = img_name[img_name.rfind('/') :][1 :]
        annot_file_path = os.path.join(self.annotations_dir,img_name[: img_name.rfind('/')]+'_labels',label_id+'.xml')
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # box coordinates for xml files are extracted
        for member in root.findall("object"):
            # bounding box
            xmin = int(member.find("bndbox").find("xmin").text)
            xmax = int(member.find("bndbox").find("xmax").text)

            ymin = int(member.find("bndbox").find("ymin").text)
            ymax = int(member.find("bndbox").find("ymax").text)

            if xmax > xmin and ymax > ymin:
                labels.append(self.labels.index(img_name[: img_name.rfind('/')]))
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

    def load_dataset(self, image_dir, annotations_dir, csv_file):
        labels = []
        annotations = []
        data = []
        ids = []
        for i in os.listdir(image_dir):
            for j in os.listdir(image_dir + i):
                ids.append(i + '/' +j)

        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                if i in split_list:
                    ids_final.append(i)
            ids = ids_final
            
        for img in ids:
            label_id = img[img.rfind('/') :][1 :]
            label_id_final = label_id[: label_id.rfind('.png')]
            annot_file_path = os.path.join(annotations_dir,img[: img.rfind('/')]+'_labels',label_id_final+'.xml')
            tree = et.parse(annot_file_path)
            root = tree.getroot()

            # box coordinates for xml files are extracted
            has_box = False
            
            for member in root.findall("object"):

                xmin = int(member.find("bndbox").find("xmin").text)
                xmax = int(member.find("bndbox").find("xmax").text)

                ymin = int(member.find("bndbox").find("ymin").text)
                ymax = int(member.find("bndbox").find("ymax").text)

                if xmax > xmin and ymax > ymin:
                    has_box = True
                    annotations.append({"image_id": img, "label": img[: img.rfind('/')]})

        labels = ["no-data","ship","coast_ship","multi"]

        for img in ids:
            name = img[: img.rfind('.png')]
            data.append(name)

        return labels, annotations, data

    def show_image(self, index, show_title=False):
        # plot the image and bboxes
        # Bounding boxes are defined as follows: x-min y-min width height
        img, target = self[index]
        fig = plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")

        ax = plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.axis("off")
        for box, label in zip(target["boxes"], target["labels"]):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor="violet", facecolor="none"
            )

            # Draw the bounding box on top of the image
            ax.add_patch(rect)
            
            ax.annotate(
                self.labels[label],
                (box[0] + 15, box[1] - 20),
                color="violet",
                fontsize=12,
                ha="center",
                va="center",
            )
            
        plt.tight_layout()
        plt.show()
        return fig

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").count()
        df_count = df_count.iloc[::-1,:].reset_index()
        df_count.columns = ["Label", "Count"]
        return df_count
    
    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=objects_count, x="Label", y="Count", ax=ax)
        ax.set_title(
            "Number of instances for {} dataset".format(self.get_name()), pad=20, fontsize=18
        )
        return fig