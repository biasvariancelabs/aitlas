import torch

from torchvision import transforms
from ..base import BaseTransforms


class ResizeToTensorNormalizeRGB(BaseTransforms):

    configurables = ["bands10_mean", "bands10_std"]

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.bands10_mean = kwargs["bands10_mean"]
        self.bands10_std = kwargs["bands10_std"]

    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((224, 224)),
            transforms.Normalize(self.bands10_mean, self.bands10_std)
        ])
        return data_transforms(sample)


class ToTensorResizeRandomCropFlipHV(BaseTransforms):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        return data_transforms(sample)


class ToTensorResizeCenterCrop(BaseTransforms):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
        ])

        return data_transforms(sample)


class ToTensorResize(BaseTransforms):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((224, 224)),
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
