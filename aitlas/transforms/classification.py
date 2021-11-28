import numpy as np

from torchvision import transforms
from ..base import BaseTransforms


class ResizeCenterCropFlipHVToTensor(BaseTransforms):
    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
        ])

        return data_transforms(sample)


class ResizeFlipHV(BaseTransforms):
    def __call__(self, sample):
        bands10, multihots = sample

        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        return data_transforms(np.uint8(bands10)), multihots


class ResizeCenterCropToTensor(BaseTransforms):
    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        return data_transforms(sample)


class Resize(BaseTransforms):
    def __call__(self, sample):
        bands10, multihots = sample

        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
        ])

        return data_transforms(np.uint8(bands10)), multihots


class ConvertToRGBResizeCenterCropToTensor(BaseTransforms):
    def __call__(self, sample):
        sample = sample[:, :, :3]
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        return data_transforms(sample)
