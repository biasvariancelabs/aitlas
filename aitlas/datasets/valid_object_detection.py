import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionCocoSchema
from .object_detection import BaseObjectDetectionDataset


'''
VALID stands for Virtual Aerial Image Dataset. It consits of 6690 high-resolution images, all annotated 
with panoptic segmentation on 30 categories, object detection with the oriented bounding box, and binocular 
depth maps, collected in 6 different virtual scenes and 5 various ambient conditions (sunny, dusk, night, 
snow and fog). This subset considers the object detection task with horizontal bounding boxes.
'''


class VALIDObjectDetectionDataset(BaseObjectDetectionDataset):

    url = ["https://drive.google.com/drive/folders/1nKQqR92NqNEWj-KXvxAKekFdQikrotLj"]
    schema = FAIREOObjectDetectionCocoSchema
    name = "VALID for object detection"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)

        # load the config
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file

        # Initialize an O(1) lookup dictionary for annotations
        self.img_to_annotations = {}

        # load the data
        self.labels, self.data, self.annotations, self.file_names = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )

    def __getitem__(self, index):
        # Safely loop to find a valid sample without causing an IndexError
        for _ in range(10): # skip cases where the amount of boxes and labels differ
            img_data = self.file_names[index]

            # reading the images and converting them to correct size and color
            image_path = os.path.join(self.data_dir, img_data)
            image = image_loader(image_path) / 255.0
            if np.shape(image)[2] == 4:
                image = image[:, :, 0:3]

            # O(1) fetch from memory
            detections = self.img_to_annotations.get(img_data, [])

            boxes = []
            labels = []
            for annotation in detections:
                labels.append(annotation["category_id"]) 
                bbox = annotation["hbbox"]

                if len(bbox) > 0:
                    # bounding box
                    xmin = bbox[0]
                    xmax = bbox[0] + bbox[2]

                    ymin = bbox[1]
                    ymax = bbox[1] + bbox[3]
                    
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "iscrowd": iscrowd
            }

            image, target = self.apply_transformations(image, target)

            # Check if valid. If so, break and return. 
            if len(target["boxes"]) > 0 and len(target["boxes"]) == len(target["labels"]):
                break
            
            # Move to the next index safely, wrapping around to avoid out-of-bounds errors
            index = (index + 1) % len(self.file_names)

        return image, target
        
    def load_dataset(self, data_dir, annotation_dir, csv_file):
        annotations = []
        data = []
        file_names = []
        ids = set() # Use a SET for O(1) lookups instead of a List
        
        for scene in os.listdir(os.path.join(data_dir, "images")):
            for altitude in os.listdir(os.path.join(data_dir, "images", scene)):
                for img_name in os.listdir(os.path.join(data_dir, "images", scene, altitude)):
                    ids.add(os.path.join(scene, altitude, img_name))
        
        # BVLabs provides csv files with a 60/20/20 split and a stratification based on scene type and collection altitude 
        if csv_file: 
            ids = set() # Reset as a set
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    scene, altitude, img_name = row[0], row[1], row[2]
                    ids.add(os.path.join(scene, altitude, img_name))

        labels = ["no-data","tree","plant","road","pavement","land","water","pool","ice","stone","pier rubble","bridge",
                  "sign","small vehicle","large vehicle","building","animal","person","chair","fence","garbage bin",
                  "low obstacle","telegraph pole","traffic light","bus stop","lamp","high obstacle","tunnel","ship",
                  "plane","harbor"]

        for json_file in os.listdir(annotation_dir):
            json_path = os.path.join(annotation_dir, json_file)
            with open(json_path, "r") as f:
                coco = json.load(f)
                
            name = coco["file_name"][7:]
            
            # Instant O(1) lookup
            if name in ids:
                # create data
                dataf = [
                    {
                        "id": coco["id"],
                        "annotations": [],
                        "file_name": coco["file_name"]
                    }
                ]
                data_inverted = {x["id"]: i for i, x in enumerate(dataf)}
                annotationsf = coco["detection"]

                # Initialize empty list for this specific image
                self.img_to_annotations[coco["file_name"]] = []

                # create index and annotations
                for annotation in annotationsf:
                    bbox = [max(coor, 0) for coor in annotation["hbbox"]]
                    annotation["hbbox"] = bbox
                    
                    key = data_inverted[coco["id"]]
                    dataf[key]["annotations"].append(annotation)
                    annotations.append(annotation)
                    
                    # Pre-cache the formatted annotation for __getitem__
                    self.img_to_annotations[coco["file_name"]].append(annotation)

                data.append(dataf)
                file_names.append(coco["file_name"])

        return labels, data, annotations, file_names

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df = df.drop(['id','segmentation','hbbox','obbox'], axis=1)
        df_count = df.groupby("category_name").count()
        df_count = df_count.iloc[:, ::-1].reset_index()
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