"""Contains joint transforms for images and label masks."""
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2

from ..base import BaseTransforms


class FlipHVRandomRotate(BaseTransforms):
    """
    A class that applies flipping, random rotation, and shift-scale-rotation transformations to image and mask pairs.
  
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Tuple of input image and mask
        :type sample: tuple
        :return: Transformed image and mask
        :rtype: tuple
        """
        image, mask = sample
        image = np.asarray(image)
        mask = np.asarray(mask)
        data_transforms = A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.9,
                    border_mode=cv2.BORDER_REFLECT,
                ),
            ]
        )
        transformed = data_transforms(image=image, mask=mask)

        return transformed["image"], transformed["mask"]


class FlipHVToTensorV2(BaseTransforms):
    """
    A class that applies resizing, flipping, and tensor conversion to images with bounding boxes and labels.
    
    """

    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Tuple of input image and target (bounding boxes and labels)
        :type sample: tuple
        :return: Transformed image and target
        :rtype: tuple
        """
        image, target = sample
        data_transforms = A.Compose(
            [
                A.Resize(480, 480),
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.5),
                ToTensorV2(p=1.0),
            ],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )

        transformed = data_transforms(
            image=image, bboxes=target["boxes"], labels=target["labels"]
        )
        target["boxes"] = torch.Tensor(transformed["bboxes"])

        return transformed["image"], target


class ResizeToTensorV2(BaseTransforms):
    """
    A class that applies resizing and tensor conversion to images with bounding boxes and labels.
    
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Tuple of input image and target (bounding boxes and labels)
        :type sample: tuple
        :return: Transformed image and target
        :rtype: tuple
        """
        image, target = sample
        data_transforms = A.Compose(
            [A.Resize(480, 480), ToTensorV2(p=1.0)],
            bbox_params={"format": "pascal_voc", "label_fields": ["labels"]},
        )

        transformed = data_transforms(
            image=image, bboxes=target["boxes"], labels=target["labels"]
        )
        target["boxes"] = torch.Tensor(transformed["bboxes"])

        return transformed["image"], target


class Resize(BaseTransforms):
    """
    A class that applies resizing to images.
    """
    def __call__(self, sample):
        """
        Apply the transformation to the input sample.

        :param sample: Input image
        :type sample: numpy.ndarray
        :return: Transformed image
        :rtype: numpy.ndarray
        """
        data_transforms = A.Compose([A.Resize(480, 480)])

        transformed = data_transforms(image=sample)

        return transformed["image"]
