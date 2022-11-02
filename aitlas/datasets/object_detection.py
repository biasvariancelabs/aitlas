import csv
import json
import os
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
from torchmetrics.detection import MeanAveragePrecision

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import ObjectDetectionDatasetSchema


class ObjectDetectionCocoDataset(BaseDataset):
    """This is a skeleton object detection dataset following the Coco format"""

    schema = ObjectDetectionDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the config
        self.data_dir = self.config.data_dir
        self.json_file = self.config.json_file

        # TODO: might wanna check this
        self.width = 224
        self.height = 224

        # load the data
        self.labels, self.data, self.annotations = self.load_dataset(
            self.data_dir, self.json_file
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_data = self.data[index]

        # reading the images and converting them to correct size and color
        img = cv2.imread(img_data["file_name"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annotations = img_data["annotations"]
        boxes = []
        labels = []

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

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

                xmin_corr = (xmin / wt) * self.width
                xmax_corr = (xmax / wt) * self.width
                ymin_corr = (ymin / ht) * self.height
                ymax_corr = (ymax / ht) * self.height

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
        target["image_id"] = img_data["id"]

        self.transform = A.Compose(
            [ToTensorV2(p=1.0)],
            bbox_params={"format": "coco", "label_fields": ["labels"]},
        )
        if self.transform:
            sample = self.transform(
                image=img_res, bboxes=target["boxes"], labels=labels
            )

            img_res = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])
        # transform = transforms.ToTensor()
        # img_res = transform(img_res)
        print(target["boxes"].shape)
        return img_res, target

    def __len__(self):
        return len(self.data)

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def get_labels(self):
        return self.labels

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

    def show_image(self, index):
        label = self.labels[self[index][1]]
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with label {label}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def show_batch(self, size, show_title=True):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])
            axes.set_title(self.labels[self[image_index][1]], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure

    def load_dataset(self, data_dir=None, json_file=None):
        if json_file:
            coco = json.load(open(json_file, "r"))

            # read labels
            labels = [
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
                key = data_inverted[annotation["image_id"]]
                data[key]["annotations"].append(annotation)
        else:
            raise ValueError(
                "Please provide the `json_file` path to the Coco annotation format."
            )

        return labels, data, annotations


def collate_fn(batch):
    return tuple(zip(*batch))
