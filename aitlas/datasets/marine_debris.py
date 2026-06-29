import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import csv

from ..utils import image_loader
from .schemas import FAIREOObjectDetectionYoloSchema
from .object_detection import BaseObjectDetectionDataset

'''
This dataset consists of images of marine debris which are 256 x 256 pixels in size, and labels 
which are bounding boxes with geographical coordinates. The images were obtained from PlanetScope 
optical imagery which has a spatial resolution of approximately 3 meters. In this dataset, marine 
debris consists of floating objects on the ocean surface which can belong to one or more classes, 
namely plastics, algae, sargassum, wood, and other artificial items.
'''

class NasaMarineDebrisDataset(BaseObjectDetectionDataset):
    schema = FAIREOObjectDetectionYoloSchema
    name = "Marine Debris Dataset"
    url = "https://source.coop/repositories/nasa/marine-debris"
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
        image = image_loader(os.path.join(self.data_dir, f"{img_name}.jpg")) /255.0 
        img_h, img_w = image.shape[:2]

        #annotation file
        annot_file_path = os.path.join(self.annotation_dir, f"{img_name}.npy")
        file_size = os.path.getsize(annot_file_path)

        if file_size > 0:
            annot = np.load(annot_file_path)
            boxes = []
            labels = []
            for annotation in annot:

                xmin = annotation[0]
                ymin = annotation[1]
                xmax = annotation[2]
                ymax = annotation[3]

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

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

            return self.apply_transformations(image, target)


    def load_dataset(self, data_dir, annotation_dir, csv_file):
        annotations = []
        data = []
        labels = ["no-data","marine debris"] 

        ids = list(set(os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(('.jpg', '.tif'))))

        split_list = []
        ids_final = []
        if csv_file: 
            with open(csv_file, mode ='r')as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    split_list.append(lines[0])
            for i in ids:
                i_csv = 'images/'+ i + '.tif'
                if i_csv in split_list:
                    ids_final.append(i)
            ids = ids_final

        for img in ids:

            npy_file_path = os.path.join(annotation_dir, img + '.npy')
            annot = np.load(npy_file_path)

            # box coordinates for npy files are extracted
            for annotation in annot:
                # bounding box
                xmin = annotation[0]
                ymin = annotation[1]
                xmax = annotation[2]
                ymax = annotation[3]

                if xmax > xmin and ymax > ymin:
                    annotations.append({"label": int(1)})

            data.append(img)

        return labels, annotations, data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", "marine debris", True)
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