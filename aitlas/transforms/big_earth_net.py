import torch
import cv2

from torchvision import transforms
from ..base import BaseTransforms


class ToTensorNormalizeRGB(BaseTransforms):

    configurables = ["bands10_mean", "bands10_std"]

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.bands10_mean = kwargs["bands10_mean"]
        self.bands10_std = kwargs["bands10_std"]

    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Normalize(self.bands10_mean, self.bands10_std)
        ])
        return data_transforms(sample)


class ResizeToTensorRGB(BaseTransforms):

    configurables = ["bands10_mean", "bands10_std"]

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.bands10_mean = kwargs["bands10_mean"]
        self.bands10_std = kwargs["bands10_std"]

    def __call__(self, sample):
        sample = cv2.resize(sample, (224, 224), interpolation=cv2.INTER_AREA)

        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
        ])
        return data_transforms(sample)


class NormalizeAllBands(BaseTransforms):

    configurables = ["bands10_mean", "bands10_std", "bands20_mean", "bands20_std"]

    def __init__(self, *args, **kwargs):
        BaseTransforms.__init__(self, *args, **kwargs)

        self.bands10_mean = kwargs["bands10_mean"]
        self.bands10_std = kwargs["bands10_std"]
        self.bands20_mean = kwargs["bands20_mean"]
        self.bands20_std = kwargs["bands20_std"]

    def __call__(self, input, target=None):
        bands10, bands20, multihots = input

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        for t, m, s in zip(bands20, self.bands20_mean, self.bands20_std):
            t.sub_(m).div_(s)

        return bands10, bands20, multihots


class ToTensorAllBands(BaseTransforms):
    def __call__(self, input, target=None):
        bands10, bands20, multihots = input
        return torch.tensor(bands10).permute(2, 0, 1), torch.tensor(bands20).permute(2, 0, 1), multihots
