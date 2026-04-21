"""Classes and methods for image transformations for segmentation tasks.
For semantic segmentation tasks the shape of the input is (N, 3, H, W);
The shape of the output/mask is (N, num_classes, H, W), where N is the number of images"""

import albumentations as A
from torchvision.transforms import v2
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
        data_transforms = v2.Compose([
            v2.ToImage(), # Converts numpy array to tensor
            v2.ToDtype(torch.float32, scale=False),
            v2.Pad(4)
        ])
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

    def __init__(self, size=256, **kwargs):
        """
        :param size: Desired output size. If int, square crop is made. If tuple, matches exact dimensions.
        """
        super().__init__(**kwargs)
        
        # Handle both int (e.g., 224) and tuple (e.g., (224, 224)) inputs
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        # Build the transformation pipeline
        self.data_transforms = v2.Compose([
            v2.ToImage(),  # Converts numpy array to tensor
            v2.ToDtype(torch.float32, scale=False),
            v2.Resize(self.size, antialias=True),
        ])

    def __call__(self, sample):
        """
        Resizes and converts the given sample to a tensor.

        :param sample: input sample
        :type sample: tensor (or numpy array depending on pipeline step)
        :return: resized tensor
        """
        return self.data_transforms(sample)


class ResizePerChannelToTensor(BaseTransforms):
    def __init__(self, size=256, **kwargs):
        """
        :param size: Desired output size. If int, square crop is made. If tuple, matches exact dimensions.
        """
        super().__init__(**kwargs)
        
        # Handle both int (e.g., 224) and tuple (e.g., (224, 224)) inputs
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        # Build the transform pipeline once during initialization
        self.data_transforms = v2.Compose([
            v2.ToImage(),  # Converts numpy array to tensor
            v2.ToDtype(torch.float32, scale=False),
            v2.Resize(self.size, antialias=True),
        ])

    def __call__(self, sample):
        """Applies resize transformations per channel. This is useful for multichannel images. In torchvision transforms v2, this is automatically handled, so this class is used for legacy support only.

        :param sample: input sample (channel, h, w)
        :type sample: tensor
        :return: resized tensor
        """
        return self.data_transforms(sample)


class ResizeToTensor224(ResizeToTensor):
    """Hardcoded 224x224 wrapper for AiTLAS string configs"""
    def __init__(self, **kwargs):
        super().__init__(size=224, **kwargs)


class ResizePerChannelToTensor224(ResizePerChannelToTensor):
    """Hardcoded 224x224 wrapper for AiTLAS string configs"""
    def __init__(self, **kwargs):
        super().__init__(size=224, **kwargs)


class ResizeToTensor120(ResizeToTensor):
    """Hardcoded 120x120 wrapper for AiTLAS string configs"""
    def __init__(self, **kwargs):
        super().__init__(size=120, **kwargs)


class ResizePerChannelToTensor120(ResizePerChannelToTensor):
    """Hardcoded 120x120 wrapper for AiTLAS string configs"""
    def __init__(self, **kwargs):
        super().__init__(size=120, **kwargs)