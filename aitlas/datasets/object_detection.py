import json
import os
import random
from xml.etree import ElementTree as et

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ..base import BaseDataset
from ..utils import collate_fn, image_loader
from .schemas import (
    ObjectDetectionCocoDatasetSchema,
    ObjectDetectionPascalDatasetSchema,
)


class BaseObjectDetectionDataset(BaseDataset):
    """Base object detection dataset class"""

    name = "Object Detection Dataset"

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def __len__(self):
        return len(self.data)

    def apply_transformations(self, image, target):
        if self.joint_transform:
            image, target = self.joint_transform((image, target))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def get_labels(self):
        return self.labels

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

    def show_batch(self, size, show_labels=False):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )

        for axes, image_index in zip(ax.flatten(), image_indices):
            img, target = self[image_index]
            axes.imshow(img)
            for box, label in zip(target["boxes"], target["labels"]):
                x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
                rect = patches.Rectangle(
                    (x, y),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="violet",
                    facecolor="none",
                )
                # Draw the bounding box on top of the image
                axes.add_patch(rect)
                if show_labels:
                    axes.annotate(
                        self.labels[label],
                        (box[0] + 15, box[1] - 20),
                        color="violet",
                        fontsize=12,
                        ha="center",
                        va="center",
                    )
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure


class ObjectDetectionPascalDataset(BaseObjectDetectionDataset):
    schema = ObjectDetectionPascalDatasetSchema

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.image_dir = self.config.image_dir
        self.annotations_dir = self.config.annotations_dir
        self.imageset_file = self.config.imageset_file

        self.labels, self.data, self.annotations = self.load_dataset(
            self.imageset_file, self.annotations_dir
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.image_dir, f"{img_name}.jpg")) / 255.0

        # annotation file
        annot_file_path = os.path.join(self.annotations_dir, f"{img_name}.xml")
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

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        # image_id
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def load_dataset(self, imageset_file, data_dir):
        labels = []
        annotations = []
        data = [f.strip() for f in open(imageset_file, "r").readlines()]
        to_remove = []
        for img in data:
            annot_file_path = os.path.join(data_dir, f"{img}.xml")
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

            if not has_box:
                to_remove.append(img)

        for img in to_remove:
            data.remove(img)

        labels = [None] + list(sorted(set(labels)))

        return labels, data, annotations

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").count().reset_index()
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


class ObjectDetectionCocoDataset(BaseObjectDetectionDataset):
    """This is a skeleton object detection dataset following the Coco format"""

    schema = ObjectDetectionCocoDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the config
        self.data_dir = self.config.data_dir
        self.json_file = self.config.json_file
        self.add_for_background = self.config.hardcode_background

        # load the data
        self.labels, self.data, self.annotations = self.load_dataset(
            self.data_dir, self.json_file
        )

    def __getitem__(self, index):
        """

        :param index: Index
        :type index: int
        :return: tuple where target is a dictionary where target is index of the target class
        :rtype: tuple of (image, target)

        """
        img_data = self.data[index]

        # reading the images and converting them to correct size and color
        image = image_loader(img_data["file_name"]) / 255.0

        # annotation file
        annotations = img_data["annotations"]
        boxes = []
        labels = []

        # box coordinates for xml files are extracted and corrected for image size given
        for annotation in annotations:
            labels.append(annotation["category_id"])

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
        target["image_id"] = torch.tensor([img_data["id"]])

        return self.apply_transformations(image, target)

    def data_distribution_table(self):
        df = pd.DataFrame([self.annotations])
        df_label = pd.DataFrame(self.labels)
        df_label.rename(columns={"0": "Label"})
        df_count = df.groupby("category_id").count()
        df_count = df_count.join(df_label)["name", "id"]
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

    def load_dataset(self, data_dir=None, json_file=None):
        if json_file:
            coco = json.load(open(json_file, "r"))

            # read labels
            labels = [None]  # add none for background
            labels += [
                y["name"] for y in sorted(coco["categories"], key=lambda x: x["id"])
            ]
            # create data
            data = [
                {
                    **x,
                    **{
                        "annotations": [],
                        **{"file_name": os.path.join(data_dir, x["file_name"])},
                    },
                }
                for x in coco["images"]
            ]
            data_inverted = {x["id"]: i for i, x in enumerate(data)}
            annotations = coco["annotations"]

            # create index and annotations
            for annotation in annotations:
                if self.add_for_background:
                    annotation["category_id"] = (
                        annotation["category_id"] + 1
                    )  # increase the label number by 1 because of the hardcoded background
                bbox = []
                for coor in annotation["bbox"]:
                    bbox.append(max(coor, 0))
                annotation["bbox"] = bbox
                key = data_inverted[annotation["image_id"]]
                data[key]["annotations"].append(annotation)
        else:
            raise ValueError(
                "Please provide the `json_file` path to the Coco annotation format."
            )

        # eliminate empty annotations
        to_delete = []
        for key, d in enumerate(data):
            if len(data[key]["annotations"]) == 0:
                to_delete.append(key)

        for key in sorted(to_delete, reverse=True):
            del data[key]

        return labels, data, annotations
