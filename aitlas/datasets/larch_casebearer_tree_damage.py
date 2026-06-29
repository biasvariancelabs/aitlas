import os
from xml.etree import ElementTree as et
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv

from aitlas.utils import image_loader
from aitlas.datasets.schemas import FAIREOObjectDetectionPascalSchema
from aitlas.datasets.object_detection import BaseObjectDetectionDataset

'''
The dataset was created to monitor forest areas affected by the larch casebearer, which is a moth that mainly 
attacks larch trees and has caused significant damage in Sweden. The original dataset contains 1543 images taken 
from two drone flying occasions over five affected areas in Västergötland, Sweden. The data set is structured in 
10 batches, numbered 1 to 10. This subset of the dataset contains the batches 1 to 5, with labels providing the 
damage level of the Larch trees.
'''


class LarchCasebearerTreeDamageDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionPascalSchema
    name = "Larch Casebearer - Tree Damage"
    url = "https://lila.science/datasets/forest-damages-larch-casebearer/"

    # labels: 0 index is reserved for background
    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.image_dir = self.config.image_dir
        self.annotations_dir = self.config.annotations_dir
        self.csv_file = self.config.csv_file

        self.labels, self.annotations, self.data = self.load_dataset(
            self.annotations_dir, self.image_dir, self.csv_file
        )

    def __getitem__(self, index):
        img_name = self.data[index]
        image = image_loader(os.path.join(self.image_dir, f"{img_name}.JPG")) / 255.0

        # annotation file
        annot_file_path = os.path.join(self.annotations_dir, f"{img_name}.xml")
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # box coordinates for xml files are extracted
        for member in root.findall("object"):
            try:
                # bounding box
                xmin = int(member.find("bndbox").find("xmin").text)
                xmax = int(member.find("bndbox").find("xmax").text)

                ymin = int(member.find("bndbox").find("ymin").text)
                ymax = int(member.find("bndbox").find("ymax").text)

                if xmax > xmin and ymax > ymin:
                    labels.append(self.labels.index(member.find("damage").text))
                    boxes.append([xmin, ymin, xmax, ymax])
            except:
                None # the object has no label

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

    def load_dataset(self, annotations_dir, image_dir, csv_file):
        labels = []
        annotations = []
        data = []

        ids = os.listdir(image_dir)
        split_list = []
        ids_final = []
        if csv_file: # contains the list of images for training, validation and testing
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'images/'+ i
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        for img in ids:
            annot_file_path = os.path.join(annotations_dir, img[: img.rfind('.JPG')]+'.xml')

            tree = et.parse(annot_file_path)
            root = tree.getroot()

            # box coordinates for xml files are extracted
            for member in root.findall("object"):
                try:
                    label = member.find('damage').text.strip()
                    labels.append(label)

                    xmin = int(member.find("bndbox").find("xmin").text)
                    xmax = int(member.find("bndbox").find("xmax").text)

                    ymin = int(member.find("bndbox").find("ymin").text)
                    ymax = int(member.find("bndbox").find("ymax").text)

                    if xmax > xmin and ymax > ymin:
                        annotations.append({"image_id": img, "label": label})
                except:
                    None #the object has no label

            name = img[: img.rfind('.JPG')]
            data.append(name)

        labels = [None] + list(sorted(set(labels)))

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
                (x, y), width, height, linewidth=2, edgecolor="white", facecolor="none"
            )

            # Draw the bounding box on top of the image
            ax.add_patch(rect)
            ax.annotate(
                self.labels[label],
                (box[0] + 15, box[1] - 20),
                color="white",
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
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig