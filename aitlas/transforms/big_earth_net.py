"""Contains classes for image transformations specific for Big Earth Net dataset."""
import torch

from torchvision import transforms
from ..base import BaseTransforms


class ResizeToTensorNormalizeRGB(BaseTransforms):
    """
    A class that applies resizing, tensor conversion, and normalization to RGB images.

    """

    configurables = ["bands10_mean", "bands10_std"]

    def __init__(self, *args, **kwargs):
        """
        Initialize the class with the given mean and standard deviation for normalization.

        :param bands10_mean: Mean values for the RGB bands
        :type bands10_mean: list
        :param bands10_std: Standard deviation values for the RGB bands
        :type bands10_std: list
        """
        super().__init__(self, *args, **kwargs)

        self.bands10_mean = kwargs["bands10_mean"]
        self.bands10_std = kwargs["bands10_std"]

    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: PIL.Image.Image
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((224, 224)),
            transforms.Normalize(self.bands10_mean, self.bands10_std)
        ])
        return data_transforms(sample)


class ToTensorResizeRandomCropFlipHV(BaseTransforms):
    """
    A class that applies resizing, tensor conversion, random cropping, and random flipping to images.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: PIL.Image.Image
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        return data_transforms(sample)


class ToTensorResizeCenterCrop(BaseTransforms):
    """
    A class that applies resizing, tensor conversion, and center cropping to images.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: PIL.Image.Image
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
        ])

        return data_transforms(sample)


class ToTensorResize(BaseTransforms):
    """
    A class that applies resizing and tensor conversion to images.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: PIL.Image.Image
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
            transforms.Resize((224, 224)),
        ])
        return data_transforms(sample)


class NormalizeAllBands(BaseTransforms):
    """
    A class that applies normalization to all bands of the input.

    """

    configurables = ["bands10_mean", "bands10_std", "bands20_mean", "bands20_std"]

    def __init__(self, *args, **kwargs):
        """
        Initialize the class with the given mean and standard deviation for normalization.

        :param bands10_mean: Mean values for the bands10
        :type bands10_mean: list
        :param bands10_std: Standard deviation values for the bands10
        :type bands10_std: list
        :param bands20_mean: Mean values for the bands20
        :type bands20_mean: list
        :param bands20_std: Standard deviation values for the bands20
        :type bands20_std: list
        """
        BaseTransforms.__init__(self, *args, **kwargs)

        self.bands10_mean = kwargs["bands10_mean"]
        self.bands10_std = kwargs["bands10_std"]
        self.bands20_mean = kwargs["bands20_mean"]
        self.bands20_std = kwargs["bands20_std"]

    def __call__(self, input, target=None):
        """
        Apply the transformation to the input bands.

        :param input: List of input bands
        :type sample: list
        :return: Normalized bands
        :rtype: list
        """
        bands10, bands20, multihots = input

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        for t, m, s in zip(bands20, self.bands20_mean, self.bands20_std):
            t.sub_(m).div_(s)

        return bands10, bands20, multihots


class ToTensorAllBands(BaseTransforms):
    """
    A class for converting all bands (list) to tensors.

    """

    def __call__(self, input, target=None):
        bands10, bands20, multihots = input
        return torch.tensor(bands10).permute(2, 0, 1), torch.tensor(bands20).permute(2, 0, 1), multihots
