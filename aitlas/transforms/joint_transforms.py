"""Contains joint transforms for images and label masks."""

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2

from ..base import BaseTransforms


def ensure_numpy_hwc(data):
    """
    Helper to convert input (Tensor, PIL, or ndarray) to a NumPy HWC array.
    Albumentations requires (Height, Width, Channels).
    """
    if isinstance(data, torch.Tensor):
        # Move to CPU and handle potential CHW -> HWC swap
        data = data.detach().cpu().numpy()
        if data.ndim == 3 and data.shape[0] in [1, 3, 4]:
            data = np.transpose(data, (1, 2, 0))
    return np.asarray(data)


def ensure_numpy_target(target_field):
    """Safely converts target tensors (boxes/labels) to numpy for Albumentations."""
    if isinstance(target_field, torch.Tensor):
        return target_field.detach().cpu().numpy()
    return target_field


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
        image = ensure_numpy_hwc(image)
        mask = ensure_numpy_hwc(mask)
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

        # Ensure all inputs are compatible with Albumentations (NumPy)
        image = ensure_numpy_hwc(image)
        boxes = ensure_numpy_target(target["boxes"])
        labels = ensure_numpy_target(target["labels"])

        data_transforms = A.Compose(
            [
                A.Resize(480, 480),
                A.HorizontalFlip(0.5),
                A.VerticalFlip(0.5),
                ToTensorV2(p=1.0),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        transformed = data_transforms(image=image, bboxes=boxes, labels=labels)

        # Update target dict with transformed values
        target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)

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

        # Ensure all inputs are compatible with Albumentations (NumPy)
        image = ensure_numpy_hwc(image)
        boxes = ensure_numpy_target(target["boxes"])
        labels = ensure_numpy_target(target["labels"])

        data_transforms = A.Compose(
            [A.Resize(480, 480), ToTensorV2(p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

        transformed = data_transforms(image=image, bboxes=boxes, labels=labels)

        target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)

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
        image = ensure_numpy_hwc(sample)
        data_transforms = A.Compose([A.Resize(480, 480)])

        transformed = data_transforms(image=image)

        return transformed["image"]
