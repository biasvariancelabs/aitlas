import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import json
import numpy as np
import csv
from xml.etree import ElementTree as et
import fnmatch

from ..utils import image_loader
from .schemas import RarePlanesSchema
from .object_detection import BaseObjectDetectionDataset


'''
RarePlanes is a unique computer vision dataset combining synthetic and real satellite imagery of aircraft. 
It consists of 253 Maxar WorldView-3 satellite scenes, paired with 50000 synthetic satellite images generated 
using AI.Reverie's simulation platform.
'''


class RarePlanesDataset(BaseObjectDetectionDataset):
    schema = RarePlanesSchema
    name = "RarePlanes"
    url = "https://www.iqt.org/library/the-rareplanes-dataset"
    labels = ["None", "Airplane"]

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.annotation_dir = self.config.annotation_dir
        self.task = self.config.task
        self.imagery = self.config.imagery
        self.selection = self.config.selection
        self.annotations, self.data = self.load_dataset(
            self.data_dir, self.annotation_dir, self.csv_file
        )


    def __getitem__(self, index):
        for i in np.arange(2): # skip images which have a different number of labels and bboxes after transformations
            img_name = self.data[index][0]
            original_split = self.data[index][1]
            data_type = self.data[index][2]

            if data_type == "real":
                if self.csv_file == None: # original splits
                    data_type = ''

                image = image_loader(os.path.join(self.data_dir, data_type, original_split, 'PS-RGB_tiled', f"{img_name}.png")) / 255 # 3 bands image
                img_h, img_w = image.shape[:2]    

                # annotations are in geojson files for real images
                annot_file_path = os.path.join(self.data_dir, data_type, original_split, 'geojson_aircraft_tiled', f"{img_name}.geojson")

                boxes = []
                labels = []

                with open(annot_file_path, "r") as f:
                    annot_json = json.load(f)

                for item in annot_json["features"]:
                    points_array = np.array(item['geometry']['coordinates'][0][:-1])
                    lon_coords = points_array[:, 0]  # All longitude values
                    lat_coords = points_array[:, 1]  # All latitude values

                    xmin, ymin, xmax, ymax = self.convert_coords(img_name, data_type, original_split, lon_coords, lat_coords, img_w, img_h)

                    # Ensure the box is within image boundaries
                    xmin, xmax = max(0, xmin), min(img_w, xmax)
                    ymin, ymax = max(0, ymin), min(img_h, ymax)

                    if xmax > xmin and ymax > ymin:
                        labels.append(int(1))  # Airplane class
                        boxes.append([xmin, ymin, xmax, ymax])

            elif data_type == "synthetic":
                if self.csv_file == None: # original splits
                    data_type = ''
            
                image = image_loader(os.path.join(self.data_dir, data_type, original_split, 'images', f"{img_name}.png")) / 255 # 4 bands image
                if self.selection == "rgb":
                    image = image[:,:,:3]

                img_h, img_w = image.shape[:2]    

                # annotations are in xml files for synthetic images
                annot_file_path = os.path.join(self.data_dir, data_type, original_split, 'xmls', f"{img_name}.xml")

                boxes = []
                labels = []

                tree = et.parse(annot_file_path)
                root = tree.getroot()

                # box coordinates for xml files are extracted
                for member in root.findall("object"):
                    # bounding box
                    xmin = int(member.find("bndbox2D").find("xmin").text)
                    xmax = int(member.find("bndbox2D").find("xmax").text)

                    ymin = int(member.find("bndbox2D").find("ymin").text)
                    ymax = int(member.find("bndbox2D").find("ymax").text)

                    if xmax > xmin and ymax > ymin:
                        labels.append(int(1)) # Airplane class
                        boxes.append([xmin, ymin, xmax, ymax])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # suppose all instances are not crowd
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}

            image, target = self.apply_transformations(image, target)

            if len(target["boxes"]) != len(target["labels"]):
                index = index + 1
            else:
                break

        return image, target


    def load_dataset(self, data_dir, annotation_dir, csv_file):
        # original splits
        if csv_file == None: 
            data = []
            split = self.task
            if fnmatch.fnmatch(data_dir, "*real*"):
                data_type = "real"
                annotation_dir = os.path.join(data_dir, split, 'geojson_aircraft_tiled')
                ids = list(os.path.splitext(f)[0] for f in os.listdir(annotation_dir) if f.endswith('.geojson'))
            elif fnmatch.fnmatch(data_dir, "*synthetic*"):
                data_type = "synthetic"
                annotation_dir = os.path.join(data_dir, split, 'xmls')
                ids = list(os.path.splitext(f)[0] for f in os.listdir(annotation_dir) if f.endswith('.xml'))
            for annot in ids:
                data.append((annot, split, data_type))

            # load the annotations
            annotations = []
            for annot in data:
                if annot[2] == "real":
                    annot_path = os.path.join(data_dir, annot[1], 'geojson_aircraft_tiled', f'{annot[0]}.geojson')
                    with open(annot_path, "r") as f:
                        js = json.load(f)
                    for feat in js["features"]:
                        annotations.append({"label": int(1)})
                elif annot[2] == "synthetic":
                    annot_path = os.path.join(data_dir, annot[1], 'xmls', f'{annot[0]}.xml')
                    tree = et.parse(annot_path)
                    root = tree.getroot()

                    num_objects = root.find('num_object_mask_objects').text
                    num_objects = int(num_objects)

                    for n in np.arange(num_objects):
                        annotations.append({"label": int(1)})

        # new 60/20/20 splits provided by BVLabs in csv files
        if csv_file:
            data = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    image_id, original_split, data_type = row[0], row[1], row[2]
                    if self.imagery == 'real' and data_type != 'real':
                        continue
                    if self.imagery == 'synthetic' and data_type != 'synthetic':
                        continue
                    data.append((image_id, original_split, data_type))

            # load the annotations
            annotations = []
            for annot in data:
                if annot[2] == "real":
                    annot_path = os.path.join(data_dir, annot[2], annot[1], 'geojson_aircraft_tiled', f'{annot[0]}.geojson')
                    with open(annot_path, "r") as f:
                        js = json.load(f)
                    for feat in js["features"]:
                        annotations.append({"label": int(1)})
                elif annot[2] == "synthetic":
                    annot_path = os.path.join(data_dir, annot[2], annot[1], 'xmls', f'{annot[0]}.xml')
                    tree = et.parse(annot_path)
                    root = tree.getroot()

                    num_objects = root.find('num_object_mask_objects').text
                    num_objects = int(num_objects)

                    for n in np.arange(num_objects):
                        annotations.append({"label": int(1)})

        return annotations, data
    
    
    def convert_coords(self, img_name, data_type, original_split, lon_coords, lat_coords, img_w, img_h):
        """
        Convert geographical coordinates to pixel coordinates
        """
        # Parse XML file
        xml_path = os.path.join(self.data_dir, data_type, original_split, 'PS-RGB_tiled', f'{img_name}.png.aux.xml')
        tree = et.parse(xml_path)

        # Extract geotransform values
        gt_text = tree.find('.//GeoTransform').text
        gt = [float(x) for x in gt_text.replace(',', ' ').split()]

        # Convert coordinates to pixels
        pixel_coords = []
        for lon, lat in zip(lon_coords, lat_coords):
            px = int((lon - gt[0]) / gt[1])
            py = int((lat - gt[3]) / gt[5])
            pixel_coords.append((px, py))

        # Get bounding box
        x_coords = [p[0] for p in pixel_coords]
        y_coords = [p[1] for p in pixel_coords]

        # Clip to image bounds and return
        return [
            max(0, min(x_coords)),
            max(0, min(y_coords)),
            min(img_w-1, max(x_coords)),
            min(img_h-1, max(y_coords))
        ]


    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df_count = df.groupby("label").value_counts()
        df_count = pd.DataFrame(df_count).reset_index()
        df_count = df_count.drop(['label'], axis=1)
        df_count.insert(0, "Label", "Airplane", True)
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