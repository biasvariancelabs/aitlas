import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import numpy as np
import csv

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset

'''
The OIRDS project has produced a dataset with almost 1,000 labeled images suitable for developing 
automated vehicle detection algorithms. These images contain approximately 1,800 labeled targets.
For each target, the OIRDS provides over 30 annotations and over 60 statistics that describe the 
target within the context of the image. 
Note: the different subsets correspond to different locations.
'''

class OIRDSDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "OIRDS dataset"
    url = "https://sourceforge.net/projects/oirds/"

    def __init__(self, config):
        # now call the constructor to validate the schema
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.annotation_dir = self.config.annotation_dir
        self.csv_file = self.config.csv_file
        self.labels, self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )


    def __getitem__(self, index):

        img_path = self.data[index]
        image = image_loader(img_path)

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)

        img_h, img_w = image.shape[:2]
        dataset = img_path.split('/')[-2]
        file_name = img_path.split('/')[-1][: -4]
        annotation = img_path[: img_path.rfind(file_name)] + '/DataSet' + dataset[dataset.rfind('_') :][1 :] + '.xls'
        
        df = pd.read_excel(annotation)

        boxes = []
        labels = []

        for i, row in df.iterrows():

            if file_name == os.path.splitext(str(row["Image Name"]))[0]:

                polygon = str(row["Intersection Polygon"])
                polygon = polygon.strip("[]")
                points = [tuple(map(int, pair.split())) for pair in polygon.split(';')]

                if len(points) < 2:
                    continue
                else:
                    xs = [x for x, y in points]
                    ys = [y for x, y in points]

                    xmin = min(xs)
                    ymin = min(ys)
                    xmax = max(xs)
                    ymax = max(ys)

                    label_name = str(row["Mode of Target Type"])
                    label_index = self.labels.index(label_name)

                    if ymin < 0:
                        ymin = 0
                    if xmin < 0:
                        xmin = 0
                    if ymax > img_h:
                        ymax = img_h
                    if xmax > img_w:
                        xmax = img_w
                    if xmax > xmin and ymax > ymin:
                        labels.append(int(label_index))
                        boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
        }

        return self.apply_transformations(image, target)


    def load_dataset(self, data_dir, annotation_dir, csv_file):
        labels = ["no-data","VEHICLE/CAR", "VEHICLE/PICK-UP", "VEHICLE/UNKNOWN", "VEHICLE/VAN", "VEHICLE/TRUCK"]

        if csv_file == None:
            data = []
            annotations = []
            for i in os.listdir(data_dir):
                dataset = data_dir + i
                df = pd.read_excel(annotation_dir + i + '/' + i[: 7] + i[i.rfind('_') :][1 :] + '.xls')

                for f in os.listdir(dataset):
                    if f.endswith(('.tif')):
                        data.append(os.path.join(dataset, f))

                for index, row in df.iterrows():
                    label_name = str(row["Mode of Target Type"])
                    label_index = labels.index(label_name)
                    annotations.append(label_index)
            
        # BVLabs provides csv files with a 60/20/20 split and a stratification based on subset/location 
        elif csv_file: 
            data = []
            annotations = []
            excel_cache = {}
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for img_id, subset in reader:
                    data.append(os.path.join(data_dir, subset, img_id + ".tif"))

                    # cache excel file
                    if subset not in excel_cache:
                        excel_cache[subset] = pd.read_excel(os.path.join(annotation_dir, subset, subset[:7] + subset[subset.rfind("_") + 1 :] + ".xls"))

                    df = excel_cache[subset]

                    for index, row in df.iterrows():
                        if img_id == os.path.splitext(str(row["Image Name"]))[0]:
                            label_name = str(row["Mode of Target Type"])
                            label_index = labels.index(label_name)
                            annotations.append(label_index)

        return labels, annotations, data

    def data_distribution_table(self):

        df = pd.DataFrame(self.annotations, columns=["label"])
        df_count = df["label"].value_counts().reset_index()
        df_count.columns = ["Label", "Count"]
        df_count["Label"] = df_count["Label"].apply(lambda i: self.labels[i])

        return df_count


    def data_distribution_barchart(self, show_title=True):
        objects_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=objects_count, ax=ax)
        ax.set_title(
            "Number of instances for {}".format(self.get_name()), pad=20, fontsize=18
        )
        return fig