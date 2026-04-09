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
    Per-channel MinMax Normalization and transposing a given sample.
    """

    def __call__(self, sample):
        """
        Per-channel MinMax Normalization and transposing a given sample.
        :param sample: numpy array with shape (H, W, C)
        :return: normalized and transposed tensor with shape (C, H, W)
        """
        # Calculate min and max for each channel
        min_val = np.min(sample, axis=(0, 1))
        max_val = np.max(sample, axis=(0, 1))

        # Per-channel min-max normalization
        # Add a small epsilon to avoid division by zero
        denominator = max_val - min_val
        normalized_sample = (sample - min_val) / (denominator + 1e-7)

        # Handle channels where max_val equals min_val
        normalized_sample[..., denominator == 0] = 0

        # Transpose from (H, W, C) to (C, H, W) and convert to a tensor
        return torch.tensor(normalized_sample.transpose(2, 0, 1), dtype=torch.float32)


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

        # Calculate min and max for each channel
        min_val = np.min(sample, axis=(0, 1))
        max_val = np.max(sample, axis=(0, 1))

        # Per-channel min-max normalization
        # Add a small epsilon to avoid division by zero
        denominator = max_val - min_val
        normalized_sample = (sample - min_val) / (denominator + 1e-7)

        # Handle channels where max_val equals min_val
        normalized_sample[..., denominator == 0] = 0

        return torch.tensor(normalized_sample, dtype=torch.float32)


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


class RobustZScoreNormTranspose(BaseTransforms):
    """
    Applies robust Z-score normalization to each channel, clips to 1st and 99th percentiles,
    and transposes the sample.
    """

    def __call__(self, sample):
        """
        Applies robust Z-score normalization and transposes the sample.
        :param sample: numpy array with shape (H, W, C)
        :return: torch tensor with shape (C, H, W)
        """
        # Calculate percentiles along H and W axes for each channel
        min_val, max_val = np.percentile(sample, [0.5, 99.5], axis=(0, 1))

        # Clip the sample to the calculated percentile values
        clipped_sample = np.clip(sample, min_val, max_val)

        # Calculate mean and std deviation for each channel of the clipped sample
        mean = np.mean(clipped_sample, axis=(0, 1))
        std = np.std(clipped_sample, axis=(0, 1))

        # Apply Z-score normalization
        # Add a small epsilon to std to avoid division by zero
        normalized_sample = (clipped_sample - mean) / (std + 1e-8)

        # Transpose from (H, W, C) to (C, H, W) and convert to a tensor
        return torch.tensor(normalized_sample.transpose(2, 0, 1), dtype=torch.float32)

class RobustMinMaxNormTranspose(BaseTransforms):
    """
    Applies robust per-channel MinMax normalization with clipping and transposes the sample.
    """

    def __call__(self, sample):
        """
        Applies robust per-channel MinMax normalization and transposes the sample.
        :param sample: numpy array with shape (H, W, C)
        :return: torch tensor with shape (C, H, W)
        """
        # Calculate 0.5 and 99.5 percentiles for each channel
        min_val, max_val = np.percentile(sample, [0.5, 99.5], axis=(0, 1))

        # Clip the sample to the calculated percentile values
        clipped_sample = np.clip(sample, min_val, max_val)

        # Calculate min and max of the clipped data for each channel
        clipped_min = np.min(clipped_sample, axis=(0, 1))
        clipped_max = np.max(clipped_sample, axis=(0, 1))

        # Per-channel min-max normalization on the clipped data
        # Add a small epsilon to avoid division by zero
        denominator = clipped_max - clipped_min
        normalized_sample = (clipped_sample - clipped_min) / (denominator + 1e-7)
        
        # Handle channels where max_val equals min_val
        normalized_sample[..., denominator == 0] = 0

        # Transpose from (H, W, C) to (C, H, W) and convert to a tensor
        return torch.tensor(normalized_sample.transpose(2, 0, 1), dtype=torch.float32)
    

class RobustMedianScalerTranspose(BaseTransforms):
    """
    1. Clips outliers (0.5% and 99.5%).
    2. Applies Robust Scaling (Median/IQR) on the clipped data.
    3. Transposes to (C, H, W).
    """

    def __call__(self, sample):
        """
        :param sample: numpy array with shape (H, W, C)
        :return: torch tensor with shape (C, H, W)
        """
        # Calculate 0.5 and 99.5 percentiles for each channel
        min_val, max_val = np.percentile(sample, [0.5, 99.5], axis=(0, 1))

        # Clip the sample to the calculated percentile values
        clipped_sample = np.clip(sample, min_val, max_val)

        # Calculate median and IQR for each channel of the clipped sample
        median = np.median(clipped_sample, axis=(0, 1))
        q25, q75 = np.percentile(clipped_sample, [25, 75], axis=(0, 1))
        iqr = q75 - q25

        # Normalize using the median and IQR, adding a small epsilon to avoid division by zero
        normalized_sample = (clipped_sample - median) / (iqr + 1e-7)

        # 4. TRANSPOSE
        return torch.tensor(normalized_sample.transpose(2, 0, 1), dtype=torch.float32)