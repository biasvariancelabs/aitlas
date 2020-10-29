from .config import Configurable
from .schemas import BaseTransformsSchema


TRANSFORMS_PARAMS = {
    "torchvision.transforms.Resize": 256,
    "torchvision.transforms.CenterCrop": 224,
}


class BaseTransforms(object):
    """Base class for implementing configurable transformations"""

    schema = BaseTransformsSchema

    configurables = None

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input, target=None):
        raise NotImplementedError("Please implement the `__call__` method")
