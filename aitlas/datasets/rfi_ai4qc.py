import json
import os
import cv2 
from xml.etree import ElementTree as et

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from aitlas.utils import image_loader
from aitlas.datasets.schemas import (
    ObjectDetectionPascalDatasetSchema,
    ObjectDetectionRfiCocoDatasetSchema
)
from aitlas.datasets.object_detection import BaseObjectDetectionDataset


'''
The RFI_AI4QC dataset contains a set of labeled RFIs (radio frequency interferences), in 3940 images. 
These interferences are caused by man-made sources and can lead to an artefact in the satellite image, 
typically a bright rectangular pattern. Bounding boxes were defined around RFI artefacts in Sentinel-1 
quick-looks (png images). This dataset can be used to train object detection algorithms to automatically 
detect these RFIs in a satellite image.
'''

url = "https://zenodo.org/records/13757181"
name = "RFI_AI4QC"

class ObjectDetectionRfiPascalDataset(BaseObjectDetectionDataset):
    schema = ObjectDetectionPascalDatasetSchema
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.image_dir = self.config.image_dir
        self.annotations_dir = self.config.annotations_dir

        self.labels, self.annotations, self.data = self.load_dataset(
            self.annotations_dir
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.image_dir, f"{img_name}.SAFE.png")) / 255.0

        # annotation file
        annot_file_path = os.path.join(self.annotations_dir, f"{img_name}_LABEL.xml")
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
                labels.append(self.labels.index(member.find("name").text))
                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}
        # image_id
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def load_dataset(self, data_dir):
        labels = []
        annotations = []
        data = []
        for img in os.listdir(data_dir):
            annot_file_path = os.path.join(data_dir, img)
            tree = et.parse(annot_file_path)
            root = tree.getroot()

            # box coordinates for xml files are extracted
            has_box = False
            for member in root.findall("object"):
                label = member.find("name").text.strip()
                labels.append(label)

                xmin = int(member.find("bndbox").find("xmin").text)
                xmax = int(member.find("bndbox").find("xmax").text)

                ymin = int(member.find("bndbox").find("ymin").text)
                ymax = int(member.find("bndbox").find("ymax").text)

                if xmax > xmin and ymax > ymin:
                    has_box = True
                    annotations.append({"image_id": img, "label": label})

        labels = list(sorted(set(labels)))

        for img in os.listdir(data_dir):
            name = img[: img.rfind('_LABEL.xml')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").count()
        df_count = df_count.iloc[::-1,:].reset_index()
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
    

class ObjectDetectionRfiCocoDataset(BaseObjectDetectionDataset):
    """This is a skeleton object detection dataset following the Coco format"""

    schema = ObjectDetectionRfiCocoDatasetSchema
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the config
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir

        # load the data
        self.labels, self.data, self.annotations, self.file_names = self.load_dataset(
            self.data_dir, self.annotation_dir
        )

    def __getitem__(self, index):
        """

        :param index: Index
        :type index: int
        :return: tuple where target is a dictionary where target is index of the target class
        :rtype: tuple of (image, target)

        """
        img_data = self.file_names[index]

        # reading the images and converting them to correct size and color
        image = image_loader(os.path.join(self.data_dir, img_data+'.png')) / 255.0

        # annotation file
        name = img_data[: img_data.rfind('.SAFE')]
        annotations_file_path = os.path.join(self.annotation_dir,name+'_LABEL.json')
        annotations = json.load(open(annotations_file_path, "r"))
        
        boxes = []
        labels = []
        image_id = []

        # box coordinates for json files are extracted
        for annotation in annotations["annotations"]:
            labels.append(annotation["category_id"]) 
            image_id.append(annotation["image_id"])

            bbox = annotation["bbox"]

            if len(bbox) > 0:
                # bounding box
                xmin = bbox[0]
                xmax = bbox[0] + bbox[2]

                ymin = bbox[1]
                ymax = bbox[1] + bbox[3]

                xmin_corr = xmin
                xmax_corr = xmax
                ymin_corr = ymin
                ymax_corr = ymax

                boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor(image_id)

        return self.apply_transformations(image, target)
        
    def load_dataset(self, data_dir=None, annotation_dir=None):
        labels = []
        annotations = []
        data = []
        file_names = []
        for json_file in os.listdir(annotation_dir):
            coco = json.load(open(annotation_dir+json_file, "r"))

            # read labels
            labels += [
                y["name"] for y in sorted(coco["categories"], key=lambda x: x["id"])
            ]
            # create data
            dataf = [
                {
                    **coco["images"],
                    **{
                        "annotations": [],
                        **{"file_name": coco["images"]["file_name"]},
                    },
                }
            ]
            data_inverted = {x["id"]: i for i, x in enumerate(dataf)}
            annotationsf = coco["annotations"]

            # create index and annotations
            for annotation in annotationsf:
                bbox = []
                for coor in annotation["bbox"]:
                    bbox.append(max(coor, 0))
                annotation["bbox"] = bbox
                key = data_inverted[annotation["image_id"]]
                dataf[key]["annotations"].append(annotation)
                annotations.append(annotation)

            data.append(dataf)
            file_names.append(coco["images"]["file_name"])

        return labels, data, annotations, file_names

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df = df.drop(['image_id', 'bbox'], axis=1)
        df_label = pd.DataFrame(self.labels)
        
        df_count = df.groupby("category_id").count()
        df_count = df_count.join(df_label)
        df_count = df_count.iloc[:, ::-1].reset_index()
        df_count = df_count.drop(['category_id'], axis=1)
        df_count.columns = ["Label", "Count"]
        
        return df_count

    def data_distribution_barchart(self):
        df_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=df_count, ax=ax)
        ax.set_title(
            "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig

    def show_samples(self):
        df = pd.DataFrame(self.annotations)
        return df.head(20)
    

class ObjectDetectionRfiYoloDataset(BaseObjectDetectionDataset):
    schema = ObjectDetectionRfiCocoDatasetSchema
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
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.SAFE.png")) / 255.0
        img_h, img_w = image.shape[:2]

        # annotation file
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}_LABEL.txt")
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
                labels.append(int(elements[0]))
                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 
        # image_id
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def load_dataset(self, data_dir, annotation_dir):
        labels = []
        annotations = []
        data = []
        for img in os.listdir(annotation_dir):
            annot_file_path = os.path.join(annotation_dir, img) 
            annot = open(annot_file_path, "r")
            annot = annot.readlines()
            
            img_name = img[: img.rfind('_LABEL')]
            image_file_path = os.path.join(data_dir, f"{img_name}.SAFE.png")
            image = cv2.imread(image_file_path)
            img_h, img_w = image.shape[:2]

            labels = ["RFI","Other Anomaly"]
            
            # box coordinates for txt files are extracted
            
            for annotation in annot:

                lines = annotation[:-1] #removes /n
                elements = lines.split(" ")
                # bounding box
                xmin = round((float(elements[1])-(float(elements[3])/2))*img_w)
                xmax = round((float(elements[1])+(float(elements[3])/2))*img_w)
    
                ymin = round((float(elements[2])-(float(elements[4])/2))*img_h)
                ymax = round((float(elements[2])+(float(elements[4])/2))*img_h)

                if xmax > xmin and ymax > ymin:
                    annotations.append({"label": int(elements[0])})

        for img in os.listdir(data_dir):
            name = img[: img.rfind('.SAFE.png')]
            data.append(name)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", ["RFI","Other Anomaly"], True)
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