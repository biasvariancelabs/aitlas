""" Contains classes for image transformations for classification datasets."""

import albumentations as A
import torch
import cv2

from torchvision import transforms
from ..base import BaseTransforms


class ResizeRandomCropFlipHVToTensor(BaseTransforms):
    """
    A class that applies resizing to (256,256), random cropping to size (224,224), random flipping, and tensor conversion to images.
    
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: numpy.ndarray
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
        ])

        return data_transforms(sample)


class ResizeCenterCropFlipHVToTensor(BaseTransforms):
    """
    A class that applies resizing to (256,256), center cropping to size (224,224), random HV flipping, and tensor conversion to images.
    
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: numpy.ndarray
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
        ])

        return data_transforms(sample)


class ResizeCenterCropToTensor(BaseTransforms):
    """
    A class that applies resizing to (256,256), center cropping to size (224,224), and tensor conversion to images.
    
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: numpy.ndarray
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        return data_transforms(sample)


class Resize1ToTensor(BaseTransforms):
    """
    A class that applies fixed resizing to (224,224) and tensor conversion to images.
    
    """
    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        return data_transforms(sample)


class GrayToRGB(BaseTransforms):
    """
    A class that converts grayscale images to RGB format  [height, width, channels].
    
    """
    # Convert numpy array from gray to rgb loaded as [height, width, channels]
    def __call__(self, sample):
        if sample.ndim == 2:
            return cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
        else:
            return sample


class ConvertToRGBResizeCenterCropToTensor(BaseTransforms):
    """
    A class that converts an image to RGB format, applies resizing to size (256,256), center cropping to size (224,224), and tensor conversion.
    
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: numpy.ndarray
        :return: Transformed image
        :rtype: torch.Tensor
        """
        sample = sample[:, :, :3]
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        return data_transforms(sample)


class RandomFlipHVToTensor(BaseTransforms):
    """
    A class that applies random flipping and tensor conversion to images.
    
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: numpy.ndarray
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # transform the image from H x W x C to C x H x W
        ])

        return data_transforms(sample)


class ComplexTransform(BaseTransforms):
    """
    A class that applies complex transformations to images and tensor conversion.

    The transformations include: 

        * resizing to (256,256), random cropping to size (224,224), 
        * random flipping (H and V) with probability 50%, 
        * random brightness and constrast with probability 75%,  
        * random blur (motion, median, gaussian, and noise) with probability 70%,
        * random distortion (optical, grid, elastic) with probability 70%,, 
        * random CLAHE with probability 70%, 
        * random HSV shift with probability 50%, 
    
    """

    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: numpy.ndarray
        :return: Transformed image
        :rtype: torch.Tensor
        """
        data_transforms = A.Compose([
            #A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            #A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            #A.Resize(image_size, image_size),
            A.Cutout(max_h_size=int(224 * 0.375), max_w_size=int(224 * 0.375), num_holes=1, p=0.7),
            #A.Normalize()
            #ToTensorV2(p=1.0)
        ])
        transformed = data_transforms(image=sample)
        transformed = torch.tensor(transformed["image"].transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return transformed





