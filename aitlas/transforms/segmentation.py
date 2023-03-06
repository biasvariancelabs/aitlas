import albumentations as A
from torchvision import transforms
import torch
import numpy as np

from ..base import BaseTransforms

# for semantic segmentation tasks the shape of the input is (N, 3, H, W)
# the shape of the output/mask is (N, num_classes, H, W), where N is the number of images


class MinMaxNormTranspose(BaseTransforms):
    def __call__(self, sample):
        return torch.tensor(sample.transpose(2, 0, 1), dtype=torch.float32) / 255


class Transpose(BaseTransforms):
    def __call__(self, sample):
        return torch.tensor(sample.transpose(2, 0, 1), dtype=torch.float32)


class MinMaxNorm(BaseTransforms):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32) / 255


class Pad(BaseTransforms):
    def __call__(self, sample):
        data_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Pad(4),
                transforms.ToTensor()
            ]
        )
        return data_transforms(sample)


class ColorTransformations(BaseTransforms):
    def __call__(self, sample):
        sample = np.asarray(sample)
        data_transforms = A.Compose([
            A.OneOf([
                A.HueSaturationValue(10, 15, 10),
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),
            ], p=0.3),
        ])
        return data_transforms(image=sample)["image"]


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

