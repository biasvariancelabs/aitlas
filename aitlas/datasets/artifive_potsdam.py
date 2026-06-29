import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import torch
import csv

from ..utils import image_loader
from .schemas import ArtifivePotsdamSchema
from .object_detection import BaseObjectDetectionDataset

'''
A benchmark dataset for vehicle detection in aerial imagery. Combining real data from the 
Potsdam dataset with artificially generated vehicle images, to study the impact of artificial 
data enrichment on machine learning performance.
'''

class ArtifiVePotsdamDataset(BaseObjectDetectionDataset):
    schema = ArtifivePotsdamSchema
    name = "ArtifiVe-Potsdam"
    url = "https://rs.ipb.uni-bonn.de/data/artifive-potsdam/index.html"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file
        self.data_selection = self.config.data_selection
        self.imagery = self.config.imagery
        self.labels, self.annot, self.data, self.original_splits, self.data_types = self.load_dataset(self.annotation_dir, self.csv_file, self.data_selection)

    def __getitem__(self, index):
        for i in np.arange(2): # skip images where number of labels and bboxes don't match after transformations

            img_name = self.data[index]

            if self.csv_file:
                original_split = self.original_splits[index]
                data_type = self.data_types[index]
                if data_type == 'real':
                    annotation_dir = (os.path.join(self.data_dir, 'patched/600x600', original_split, "annotations_training_patches.json") if original_split == 'training' else os.path.join(self.data_dir, 'patched/600x600', original_split, "annotations_test_patches.json"))
                    image = image_loader(os.path.join(self.data_dir, 'patched/600x600', original_split, f"{img_name}.jpg"))/255
                elif data_type == 'artificial':
                    annotation_dir = os.path.join(self.data_dir, 'artificial', "annotations.json")
                    image = image_loader(os.path.join(self.data_dir, 'artificial', f"{img_name}.jpg"))/255
            else:
                original_split = ''
                annotation_dir = self.annotation_dir
                image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg"))/255

            img_h, img_w = image.shape[:2]

            boxes = []
            labels = []

            with open(annotation_dir, "r") as f:
                annot_json = json.load(f)

            for item in annot_json:
                img_name_json = item['image_filename'].split(".")[0]
                if img_name_json != img_name:
                    continue

                annotations = item['annotations']['polygons']
                for annotation in annotations:
                    coords = annotation['coordinates']
                    coords = coords[0]
                    xs = [point[0] for point in coords]
                    ys = [point[1] for point in coords]

                    ymin=min(ys)
                    xmin=min(xs)
                    ymax=max(ys)
                    xmax=max(xs)

                    if ymin < 0:
                        ymin = 0
                    if xmin < 0:
                        xmin = 0
                    if ymax > img_h:
                        ymax = img_h
                    if xmax > img_w:
                        xmax = img_w
                    if xmax > xmin and ymax > ymin:
                        labels.append(int(1))
                        boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

            image, target = self.apply_transformations(image, target)

            if len(target['boxes']) != len(target['labels']):
                index = index + 1
            else:
                break

        return image, target


    def load_dataset(self, annotation_dir, csv_file, data_selection):

        data = []
        annot = []
        original_splits = []
        data_types = []
        labels = ["no-data", "car"]

        # BVLabs provides csv files with a 60/20/20 split
        if csv_file:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None)
                csv_entries = {}  # img_name -> (original_split, data_type)
                for row in reader:
                    img_name, original_split, data_type = row[0], row[1], row[2]
                    if self.imagery == 'real' and data_type != 'real':
                        continue
                    if self.imagery == 'artificial' and data_type != 'artificial':
                        continue
                    csv_entries[img_name] = (original_split, data_type)

            if self.imagery == 'real':
                annot_paths = ["patched/600x600/training/annotations_training_patches.json", "patched/600x600/test/annotations_test_patches.json"]
            elif self.imagery == 'artificial':
                annot_paths = ["artificial/annotations.json"]
            elif self.imagery == 'all':
                annot_paths = ["patched/600x600/training/annotations_training_patches.json", "patched/600x600/test/annotations_test_patches.json", "artificial/annotations.json"]

            for task in annot_paths:
                with open(self.data_dir + task, "r") as f:
                    task_annot_json = json.load(f)
                for item in task_annot_json:
                    img_name = item['image_filename'].split(".")[0]
                    if img_name not in csv_entries:
                        continue
                    annotations = item['annotations']['labels']
                    if len(annotations) == 0 and data_selection == 'labeled':
                        continue  # skip unannotated
                    original_split, data_type = csv_entries[img_name]
                    data.append(img_name)
                    original_splits.append(original_split)
                    data_types.append(data_type)
                    for vehicle in annotations:
                        annot.append({"label": int(1)})

        elif csv_file == None: # use the original splits
            data = []
            annot = []
            with open(self.annotation_dir, "r") as f:
                annot_json = json.load(f)

            for item in annot_json:
                annotations = item['annotations']['labels']
                if len(annotations) == 0 and data_selection == "all":
                    img_name = item['image_filename'].split(".")[0]
                    data.append(img_name)     
                else: # only keep images which have one or more bounding box
                    for vehicle in annotations:
                        annot.append({"label": int(1)})    
                    img_name = item['image_filename'].split(".")[0]
                    data.append(img_name)         

        return labels, annot, data, original_splits, data_types

    def data_distribution_table(self):
        df = pd.DataFrame(self.annot)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", "Car", True)
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