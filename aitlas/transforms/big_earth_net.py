import torch

from ..base import BaseTransforms


class NormalizeRGB(BaseTransforms):

    configurables = ["bands10_mean", "bands10_std"]

    def __init__(self, *args, **kwargs):
        BaseTransforms.__init__(self, *args, **kwargs)

        self.bands10_mean = kwargs["bands10_mean"]
        self.bands10_std = kwargs["bands10_std"]

    def __call__(self, input, target=None):
        bands10, multihots = input

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        return bands10, multihots


class ToTensorRGB(BaseTransforms):
    def __call__(self, input, target=None):
        bands10, multihots = input
        return torch.tensor(bands10), multihots


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
        return torch.tensor(bands10), torch.tensor(bands20), multihots
