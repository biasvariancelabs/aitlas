from ..base import BaseTransforms
import albumentations as A


class MinMaxNormTransponse(BaseTransforms):
    def __call__(self, sample):
        return sample.transpose(2, 0, 1).astype("float32") / 255


class Transponse(BaseTransforms):
    def __call__(self, sample):
        return sample.transpose(2, 0, 1).astype("float32")


class MinMaxNorm(BaseTransforms):
    def __call__(self, sample):
        return sample.astype("float32") / 255


class ColorTransformations(BaseTransforms):
    def __call__(self, sample):
        data_transforms = A.Compose([
            A.OneOf([
                A.HueSaturationValue(10, 15, 10),
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),
            ], p=0.3),
        ])
        return data_transforms(sample)
