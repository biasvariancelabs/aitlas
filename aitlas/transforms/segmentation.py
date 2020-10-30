import torchvision.transforms as transforms

from ..base import BaseTransforms


class BaseSegmentation(BaseTransforms):
    def __call__(self, input, target=None):
        return (
            input.transpose(2, 0, 1).astype("float32") / 255,
            target.transpose(2, 0, 1) if target else 0,
        )
