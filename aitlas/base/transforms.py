from .config import Configurable
from .schemas import BaseTransformsSchema


TRANSFORMS_PARAMS = {"torch.transforms.Resize": 256, "torch.transforms.CenterCrop": 224}


class BaseTransforms(object):
    """Base class for implementing configurable transformations"""

    schema = BaseTransformsSchema

    configurables = None

    def __init__(self, *args, **kwargs):
        self.transform = self.load_transforms()

    def __call__(self, input, target=None):
        raise NotImplementedError("Please implement the `__call__` method")

    def load_transforms(self):
        return []
        # raise NotImplementedError("Please implement your transformations")
