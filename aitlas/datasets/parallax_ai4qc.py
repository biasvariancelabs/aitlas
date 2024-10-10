import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from aitlas.utils import image_loader
from aitlas.datasets.schemas import (
    ObjectDetectionRfiCocoDatasetSchema
)
from aitlas.datasets.object_detection import BaseObjectDetectionRotatedBboxDataset
from aitlas.datasets.object_detection import BaseObjectDetectionDataset


'''
The Parallax_AI4QC dataset contains a set of labeled parallax artefacts, in 1764 images. This 
effect appears as a colorful pattern in images where cloud/heavy haze is present. Bounding boxes 
were defined around Parallax artefacts in Sentinel-2 true color images (jpg images). This dataset 
can be used to train object detection algorithms to automatically detect these parallax in a 
satellite image. Two sets of labels are available: rotated bounding boxes and straight bounding boxes.
'''

url = "https://zenodo.org/records/13903820"
name = "Parallax_AI4QC"


class ObjectDetectionRotatedParallaxDataset(BaseObjectDetectionRotatedBboxDataset):
    """This is a skeleton object detection dataset following the Coco format"""

    schema = ObjectDetectionRfiCocoDatasetSchema

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
        img_name = img_data
        image = image_loader(os.path.join(self.data_dir, img_name+'.jpg')) / 255.0

        # annotation file
        annotations_file_path = os.path.join(self.annotation_dir,img_name+'_LABEL.json')
        annotations = json.load(open(annotations_file_path, "r"))
        
        boxes = []
        labels = []
        image_id = []
        rotation = []

        # box coordinates for json files are extracted and corrected for image size given
        for annotation in annotations["annotations"]:
            labels.append(annotation["category_id"]) 
            image_id.append(annotation["image_id"])
            rotation.append(annotation["attributes"]["rotation"])

            bbox = annotation["bbox"]

            if len(bbox) > 0:
                # bounding box
                xmin = bbox[0]
                xmax = bbox[0] + bbox[2]

                ymin = bbox[1]
                ymax = bbox[1] + bbox[3]

                if xmax > 2196:
                    xmax = 2196
                if ymax > 2196:
                    ymax = 2196

                xmin_corr = xmin
                xmax_corr = xmax
                ymin_corr = ymin
                ymax_corr = ymax

                boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        rotation = torch.as_tensor(rotation, dtype=torch.float32)
            
        target = {}
        target["boxes"] = boxes
        target["rotation"] = rotation
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id

        image, target = self.apply_transformations(image, target)
        
        return image,target
    
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
                        **{"file_name": coco["images"]["id"]},
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
            file_names.append(coco["images"]["id"])

        return labels, data, annotations, file_names

    def test(self,index):
        img_data = self.file_names[index]
        return img_data

    def data_distribution_table(self):
        df = pd.DataFrame(self.annotations)
        df = df.drop(['image_id', 'bbox', 'attributes'], axis=1)
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
    

class ObjectDetectionStraightParallaxDataset(BaseObjectDetectionDataset):
    """This is a skeleton object detection dataset following the Coco format"""

    schema = ObjectDetectionRfiCocoDatasetSchema

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
        image = image_loader(os.path.join(self.data_dir, img_data+'.jpg')) / 255.0

        # annotation file
        name = img_data
        annotations_file_path = os.path.join(self.annotation_dir,name+'_LABEL.json')
        annotations = json.load(open(annotations_file_path, "r"))
        
        boxes = []
        labels = []
        image_id = []

        # box coordinates for json files are extracted and corrected for image size given
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

                boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["iscrowd"] = iscrowd

        data_transforms = A.Compose(
            [A.Resize(480, 480), ToTensorV2(p=1.0)],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )
    
        transformed = data_transforms(
            image=image, bboxes=target["boxes"], labels=target["labels"]
        )
        target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        image = transformed["image"]

        return image,target
        
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
            file_names.append(coco["images"]["id"])

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