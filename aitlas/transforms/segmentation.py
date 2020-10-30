import torchvision.transforms as transforms
import torch

from ..base import BaseTransforms


class BaseSegmentation(BaseTransforms):
    def __call__(self, input, target=None):
        return (
            torch.from_numpy(input.transpose(2, 0, 1).astype("float32") / 255),
            torch.from_numpy(target.transpose(2, 0, 1)) if target else 0,
        )
