import torch

from ..base import BaseTransforms


class NormalizeRGB(BaseTransforms):
    def __init__(self, bands_mean, bands_std):
        self.bands10_mean = bands_mean
        self.bands10_std = bands_std

    def __call__(self, sample):
        bands10, multihots = sample

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        return bands10, multihots


class ToTensorRGB(BaseTransforms):
    def __call__(self, sample):
        bands10, multihots = sample

        return torch.tensor(bands10), multihots


class NormalizeAllBands(BaseTransforms):
    def __init__(self, bands10_mean, bands10_std, bands20_mean, bands20_std):
        self.bands10_mean = bands10_mean
        self.bands10_std = bands10_std
        self.bands20_mean = bands20_mean
        self.bands20_std = bands20_std

    def __call__(self, sample):
        bands10, bands20, multihots = sample

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        for t, m, s in zip(bands20, self.bands20_mean, self.bands20_std):
            t.sub_(m).div_(s)

        return bands10, bands20, multihots


class ToTensorAllBands(BaseTransforms):
    def __call__(self, sample):
        bands10, bands20, multihots = sample

        return torch.tensor(bands10), torch.tensor(bands20), multihots
