import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor

from ..base import SplitableDataset
from ..utils import pil_loader
from .schemas import CrackForestSchema


class CrackForestDataset(SplitableDataset):
    schema = CrackForestSchema

    url = "https://github.com/cuilimeng/CrackForest-dataset/archive/master.zip"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        SplitableDataset.__init__(self, config)

        self.root = config.root

        # load all image files, sorting them to
        # ensure that they are aligned
        self.images = list(sorted(os.listdir(os.path.join(self.root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(self.root, "Masks"))))

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "Images", self.images[index])
        mask_path = os.path.join(self.root, "Masks", self.masks[index])

        image = pil_loader(img_path)
        mask = pil_loader(mask_path).reshape(320, 480, 1)

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

    def get_item_name(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

    def default_transform(self):
        return Compose([ToTensor()])


class ToTensorND(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]
        if len(mask.shape) == 2:
            mask = mask.reshape((1,) + mask.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,) + image.shape)
        return torch.from_numpy(image), torch.from_numpy(mask)


class Normalize(object):
    """Normalize image"""

    def __call__(self, sample):
        image, mask = sample
        return image.type(torch.FloatTensor) / 255, mask.type(torch.FloatTensor) / 255
