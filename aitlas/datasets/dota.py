from aitlas.base import BaseDataset
from .schemas import DotaDatasetSchema

import os

import random
from PIL import Image
import numpy as np

import torch

def collate_fn(batch):
    return tuple(zip(*batch))

class DotaDataset(BaseDataset):

    schema = DotaDatasetSchema

    def __init__ (self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)

        self.root = self.config.root
        self.subset = self.config.subset
        self.filter_null = self.config.filter_null

        self.mappings = {
            "plane": 1, 
            "ship": 2, 
            "storage-tank": 3, 
            "baseball-diamond": 4, 
            "tennis-court": 5, 
            "basketball-court": 6, 
            "ground-track-field": 7, 
            "harbor": 8, 
            "bridge": 9, 
            "large-vehicle": 10, 
            "small-vehicle": 11, 
            "helicopter": 12, 
            "roundabout": 13, 
            "soccer-ball-field": 14, 
            "swimming-pool": 15
        }

        self.subsample_percentage = self.config.subsample_percentage

        self.load_dataset()

    def load_dataset(self):
        # read all filenames from disk
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, self.subset, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(self.root, self.subset,  "labelTxt"))))

        print ("The number of images on disk is:", len(self.imgs))

        if self.filter_null:
            self.filter()
            print ("The number of images with at least one detectable object is:", len(self.imgs))

        # select a subset of these images for training and testing
        # the reason for this subsampling is computational complexity only
        if self.subsample_percentage != 1.0:
            num_subsampled_imgs = int(self.subsample_percentage * len(self.imgs))
            selected = random.sample(range(0,len(self.imgs)), num_subsampled_imgs)
        
            self.imgs = [self.imgs[idx] for idx in range(len(self.imgs)) if idx in selected]
            self.labels = [self.labels[idx] for idx in range(len(self.labels)) if idx in selected]

        print ("The subsampled number of images is:", len(self.imgs))

    def filter (self):
        filtered_imgs, filtered_labels = [],  []

        for (img, label) in zip(self.imgs, self.labels):
            label_path = os.path.join(self.root, self.subset, "labelTxt", label)
            file = open(label_path, "r")

            line_count = 0

            for line in file:
                 if line != "\n":
                     line_count+=1
                     break 
            
            file.close()

            if line_count:
                filtered_imgs.append(img)
                filtered_labels.append(label)

        self.imgs = filtered_imgs
        self.labels = filtered_labels

        del filtered_imgs, filtered_labels

    def __getitem__(self, idx):

        # load images and masks
        img_path = os.path.join(self.root, self.subset, "images", self.imgs[idx])
        label_path = os.path.join(self.root, self.subset, "labelTxt", self.labels[idx])
        
        img = Image.open(img_path).convert("RGB")
        
        '''
        should end up with a dictionary like this:
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        '''
        boxes = []
        labels = []

        with open(label_path, "r") as a_file:
            for line in a_file:
                line = line.strip()

                xmin, xmax, ymin, ymax = np.inf, 0, np.inf, 0

                for i in range(8):
                    value = float(line.split(" ")[i])
                    # x-axis coord
                    if i%2==0:
                        if value < xmin:
                            xmin = value
                        elif value > xmax:
                            xmax = value
                    # y-axis coord
                    else:
                        if value < ymin:
                            ymin = value
                        elif value > ymax:
                            ymax = value

                if (((xmax - xmin) <= 0.0) or ((ymax - ymin) <= 0.0)):
                    continue
                                    
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.mappings[line.split(" ")[8]])

        # if we have chosen not to filter empty images and the current images does not contain any objects 
        # append a dummy bbox and label it as background
        if (not self.filter_null) and (not len(boxes)):
            boxes.append([0, 1, 2, 3])
            labels.append(0)
    
        if not len(boxes):
            boxes.append([0, 1, 2, 3])
            labels.append(0)
        
        # convert to torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # calculate the area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])    
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # set the image index as the image identifier
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            img = self.transform(img)

        return img, target
        
    def __len__(self):
        return len(self.imgs)

    def dataloader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, 
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
                                           pin_memory = self.pin_memory,
                                           collate_fn=collate_fn)

    def get_labels(self):
        return list(self.mappings.keys())

    def get_img_names(self):
        return self.imgs

    def add_predictions (self, predictions):
        self.predictions = predictions

    def save_predictions (self, dir_path):
        # iterate over all classes and find all boxes with that class
        for class_name, class_id in self.mappings.items():
            with open(dir_path + os.sep + 'Task2_{}.txt'.format(class_name), "w") as tf:
                for img_name, predictions in zip(self.imgs, self.predictions):
                    for (box, score, label) in zip (predictions['boxes'], predictions['scores'], predictions['labels']):
                        if label == class_id:
                            tf.write ('{} {} {} {} {} {}\n'.format(img_name, score, box[0], box[1], box[2], box[3]))
                tf.close()