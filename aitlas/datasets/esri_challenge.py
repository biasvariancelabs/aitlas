from aitlas.base import BaseDataset
from .schemas import EsriChallengeDatasetSchema

import os
import xml.etree.ElementTree as ET

import random
from PIL import Image

import torch

def collate_fn(batch):
    return tuple(zip(*batch))

class EsriChallengeDataset (BaseDataset):

    schema = EsriChallengeDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)

        self.mappings = {
            "car":1,
            "pool":2
        }

        self.root = self.config.root
        self.subset = self.config.subset
        self.subsample_percentage = self.config.subsample_percentage

        self.load_dataset()
    
    def load_dataset(self):
        # read all filenames from disk
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, self.subset, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(self.root, self.subset,  "labels"))))

        print ("The number of images on disk is:", len(self.imgs))

        # select a subset of these images for training and testing
        # the reason for this subsampling is computational complexity only
        num_subsampled_imgs = int (self.subsample_percentage * len(self.imgs))
        selected = random.sample(range(0, len(self.imgs)), num_subsampled_imgs)

        self.imgs = [self.imgs[idx] for idx in range(len(self.imgs)) if idx in selected]
        self.labels = [self.labels[idx] for idx in range(len(self.labels)) if idx in selected]

        print ("The subsampled number of images is:", len(self.imgs))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.subset, "images", self.imgs[idx])
        label_path = os.path.join(self.root, self.subset, "labels", self.labels[idx])
        
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
        label_xml = ET.parse(label_path)
        root = label_xml.getroot()

        boxes = []
        labels = []
        for child_idx, child in enumerate(root):
            # then this is a bounding box object
            if child.tag == 'object':
                label = int (root[child_idx][0].text)
                xmin = float (root[child_idx][1][0].text)
                ymin = float (root[child_idx][1][1].text)
                xmax = float (root[child_idx][1][2].text)
                ymax = float (root[child_idx][1][3].text)
                boxes.append([xmin, ymin, xmax, ymax])  
                labels.append(label)
        
        # convert to torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # set the image index as the image identifier
        image_id = torch.tensor([idx])
        # calculate the area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
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