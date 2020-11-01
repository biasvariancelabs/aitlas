import torch
import torchvision.transforms as transforms

from ..base import BaseTransforms


class BaseSegmentation(BaseTransforms):
    def __call__(self, sample):
        image = sample.get("image", None)
        mask = sample.get("mask", None)
        if mask is None:
            image.transpose(2, 0, 1).astype("float32") / 255
        else:
            return (image.transpose(2, 0, 1).astype("float32") / 255, mask.transpose(2, 0, 1).astype("float32"))

