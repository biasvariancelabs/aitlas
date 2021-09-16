from ..base import BaseTransforms
import albumentations as A
import cv2


class FlipHVRandomRotate(BaseTransforms):
    def __call__(self, image, mask=None):
        data_transforms = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15,
                               p=0.9, border_mode=cv2.BORDER_REFLECT),
        ])
        return data_transforms(image), data_transforms(mask)
