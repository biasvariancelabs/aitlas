import os

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as F

category_mapping = {
    "plane":1, 
    "ship":2, 
    "storage-tank":3, 
    "baseball-diamond":4, 
    "tennis-court":5, 
    "basketball-court":6, 
    "ground-track-field":7, 
    "harbor": 8, 
    "bridge": 9, 
    "large-vehicle":10, 
    "small-vehicle":11, 
    "helicopter":12, 
    "roundabout":13, 
    "soccer-ball-field":14, 
    "swimming-pool":15
}

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

class DOTA (torch.utils.data.Dataset):
    def __init__(self, root, subset, subsample_percentage, batch_size = 4, shuffle = True, num_workers=4, filter_null = True):
        self.root = root
        self.transforms = ToTensor()
        
        # either 'train_split', 'validation_split', 'test_split'
        self.subset = subset

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.filter_null = filter_null

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, self.subset, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, self.subset,  "labelTxt"))))

        print ("The original number of images is:", len(self.imgs))

        if self.filter_null:
            self.filter()
            print ("After filtering the number of images is:", len(self.imgs))

        # select a subset of these images for training and testing
        # the reason for this subsampling is computational complexity only
        num_subsampled_imgs = int (subsample_percentage * len(self.imgs))
        selected = np.random.randint (0, high = len(self.imgs), size = num_subsampled_imgs, dtype=int)

        self.imgs = np.array(self.imgs)[selected].tolist()
        self.labels = np.array(self.labels)[selected].tolist()

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

                if ymin == ymax:
                    ymax = ymin+1
                if xmin == xmax:
                    xmax = xmin+1

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(category_mapping[line.split(" ")[8]])

        # if we have chosen not to filter empty images and the current images does not contain any objects 
        # append a dummy bbox and label it as background
        if (not self.filter_null) and (not len(boxes)):
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

        # print (boxes)
        # print (labels)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
        
    def __len__(self):
        return len(self.imgs)

    def dataloader(self):
        return torch.utils.data.DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=collate_fn)