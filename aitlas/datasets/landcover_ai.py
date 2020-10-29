import csv
import os
import numpy as np
import torch

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema

#CLASSES_TO_IDX = {"Background": 0, "Buildings": 1, "Woodlands": 2, "Water": 3}

#CLASSES_TO_IDX = {'sky': 0, 'road': 3, 'car': 8}
CLASSES_TO_IDX = {'car': 8}


class SegmentationDataset(BaseDataset):

    schema = SegmentationDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)
        self.images = []
        self.masks = []
        self.class_values = []

        self.load_dataset(self.config.root, self.config.csv_file_path)


    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], False)
        #print(image.shape, mask.shape)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')
        #print(image.shape, mask.shape)
        #image = torch.from_numpy(image.transpose(2, 0, 1).astype('float32') / 255)
        #mask = torch.from_numpy(mask.transpose(2, 0, 1))
        #print(image.size(), mask.size())


        # return image, mask
        return torch.from_numpy(image.transpose(2, 0, 1).astype('float32') / 255), torch.from_numpy(mask.transpose(2, 0, 1))

    def __len__(self):
        return len(self.images)

    def load_dataset(self, root_dir, file_path):
        self.class_values = CLASSES_TO_IDX.values()

        if not self.class_values:
            raise ValueError(
                "You need to implement the classes to index mapping for the dataset"
            )

        #with open(file_path, "r") as f:
        #    csv_reader = csv.reader(f)
        #    for index, row in enumerate(csv_reader):
        #        self.images.append(os.path.join(root_dir, row[0] + '.jpg'))
        #        self.masks.append(os.path.join(root_dir, row[0] + '_m.png'))

        ids = os.listdir(os.path.join(root_dir, 'images'))
        self.images = [os.path.join(root_dir, 'images', image_id) for image_id in ids]
        self.masks = [os.path.join(root_dir, 'masks', image_id) for image_id in ids]


