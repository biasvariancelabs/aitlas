import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import json
import csv
import numpy as np

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset


'''
High Resolution SAR Images Dataset (HRSID) is a dataset for ship detection, semantic segmentation, 
and instance segmentation tasks in high-resolution SAR images. It contains a total of 5604 high-
resolution SAR images and 16951 ship instances.
'''

class HRSIDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "HRSID"
    url = "https://github.com/chaozhong2010/HRSID"

    labels = [None]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file 

        self.labels, self.annotations, self.data = self.load_dataset(self.annotation_dir, self.csv_file)


    def __getitem__(self, index):
        for i in np.arange(2): # skip images where number of boxes and labels don't match after transformations

            img_name = self.data[index]
            image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg")) / 255.0

            boxes = []
            labels = []

            # box coordinates for json files are extracted
            for ann in self.annotations:
                if ann["image_id"] == img_name:
                    boxes.append(ann["bbox"])
                    labels.append(ann["label"])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd} 

            image, target = self.apply_transformations(image, target)

            if len(target['boxes']) != len(target['labels']):
                index = index + 1
            else:
                break

        return image, target


    def load_dataset(self, annotation_dir, csv_file):
        labels = []
        annotations = []
        data = []

        # BVLabs provides csv files with a 60/20/20 split
        if csv_file: 
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    file_name = row[0].split('/')[1]
                    base_name = file_name.rsplit('.', 1)[0]
                    data.append(base_name)

        with open(annotation_dir, 'r') as f:
            annotation_json = json.load(f)

        labels = ["no-data", "ship"]

        image_id_map = {img["id"]: img["file_name"] for img in annotation_json.get("images", [])}

        for ann in annotation_json["annotations"]:
            img_id = ann["image_id"]

            file_name = image_id_map[img_id]

            if csv_file:
                if file_name.rsplit('.', 1)[0] not in data:
                    continue 

            label_index = 1 if ann["category_id"] == 1 else 0

            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            xmax = x + w
            ymax = y + h

            annotations.append({
                "image_id": file_name.split('.')[0],
                "label": label_index,
                "bbox": [x, y, xmax, ymax]
            })

        if csv_file == None:
            for img_id, file_name in image_id_map.items():
                base_name = file_name.rsplit('.', 1)[0]
                data.append(base_name)

        return labels, annotations, data
    

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df['label_name'] = df['label'].apply(lambda x: self.labels[x])
        df_count = df.groupby("label_name").size().reset_index(name='Count')
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