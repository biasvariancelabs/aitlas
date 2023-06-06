"""Classes and methods for image transformations for segmentation tasks.
For semantic segmentation tasks the shape of the input is (N, 3, H, W);
The shape of the output/mask is (N, num_classes, H, W), where N is the number of images"""

import albumentations as A
from torchvision import transforms
import torch
import numpy as np

from ..base import BaseTransforms


class MinMaxNormTranspose(BaseTransforms):
    """
    MinMax Normalization and transposing a given sample.
    """

    def __call__(self, sample):
        """
        MinMax Normalization and transposing a given sample.

        :param sample: input sample
        :type sample: tensor
        :return: normalized and transposed tensor
        :rtype: tensor
        """
        return torch.tensor(sample.transpose(2, 0, 1), dtype=torch.float32) / 255


class Transpose(BaseTransforms):
    """
    Transposes a given sample.
    """

    def __call__(self, sample):
        """
        Transposes a given sample.

        :param sample: input sample
        :type sample: tensor
        :return: normalized and transposed tensor
        :rtype: tensor
        """
        return torch.tensor(sample.transpose(2, 0, 1), dtype=torch.float32)


class MinMaxNorm(BaseTransforms):
    """
    MinMax-Normalization of a given sample.
    """

    def __call__(self, sample):
        """
        MinMax-Normalization of a given sample.

        :param sample: input sample
        :type sample: tensor
        :return: normalized and transposed tensor
        :rtype: tensor
        """
        return torch.tensor(sample, dtype=torch.float32) / 255


class Pad(BaseTransforms):
    """
    Applies padding to a given sample.
    """

    def __call__(self, sample):
        """
        Applies padding to a given sample.

        :param sample: input sample
        :type sample: tensor
        :return: padded tensor
        :rtype: tensor
        """
        data_transforms = transforms.Compose(
            [transforms.ToPILImage(), transforms.Pad(4), transforms.ToTensor()]
        )
        return data_transforms(sample)


class ColorTransformations(BaseTransforms):
    """
    Applies a set of color transformations to a given sample.
    """

    def __call__(self, sample):
        """
        Applies color transformations to the given sample with a probability of 0.3. These include:
        * HueSaturationValue (randomly changes hue, saturation and value of the input image)
        * CLAHE (applies Contrast Limited Adaptive Histogram Equalization)
        * RandomBrightnessContrast (randomly changes brightness and contrast of the input image)

        :param sample: input sample
        :type sample: tensor
        :return: tensor after color transformations
        :rtype: tensor
        """
        sample = np.asarray(sample)
        data_transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.HueSaturationValue(10, 15, 10),
                        A.CLAHE(clip_limit=2),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
            ]
        )
        return data_transforms(image=sample)["image"]


class ResizeToTensor(BaseTransforms):
    """
    Resizes and converts a given sample to a tensor.

    """

    def __call__(self, sample):
        """
        Resizes and converts the given sample to a tensor.

        :param sample: input sample
        :type sample: tensor
        :return: resized tensor
        """
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
        """Applies resize transformations per channel. This is useful for multichannel images.

        :param sample: input sample (channel, h, w)
        :type sample: tensor
        :return: resized tensor

        """

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
