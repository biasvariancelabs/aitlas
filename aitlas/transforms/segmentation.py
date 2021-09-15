import torch
from torchvision import transforms

from ..base import BaseTransforms


class MinMaxNormTransponse(BaseTransforms):
    def __call__(self, sample):
        return sample.transpose(2, 0, 1).astype("float32") / 255


class Transponse(BaseTransforms):
    def __call__(self, sample):
        return sample.transpose(2, 0, 1).astype("float32")


class MinMaxNorm(BaseTransforms):
    def __call__(self, sample):
        return sample.astype("float32") / 255


class ResizeToTensor(BaseTransforms):
    def __call__(self, sample):
        data_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        return data_transforms(sample)


class ResizePerChannelToTensor(BaseTransforms):
    def __call__(self, sample):
        """Applies transformations per channel. Assumes this format: (channel, h, w)"""

        data_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        x = []
        # apply transformations to each channel
        for ch in sample:
            x.append(data_transforms(ch))

        # this is the multichannel transformed image (a torch tensor)
        return torch.cat(x)
