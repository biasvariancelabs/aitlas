from torchvision import transforms

from ..utils import get_class
from .schemas import BaseTransformsSchema


TRANSFORMS_PARAMS = {
    "torchvision.transforms.Resize": 256,
    "torchvision.transforms.CenterCrop": 224,
}


def load_transforms(class_names, config):
    """Loads transformation classes and make a composition of them"""
    lst_transforms = []

    if not class_names:
        return None

    # check all transformation classes
    for name in class_names:
        cls = get_class(name)  # get class
        args = TRANSFORMS_PARAMS.get(name, None)  # get params, if specified
        if args:
            transfrm = cls(args)
        else:
            if getattr(cls, "configurables", None):
                kwargs = {}
                for key in cls.configurables:
                    kwargs[key] = getattr(config, key)
                transfrm = cls(**kwargs)
            else:
                transfrm = cls()

        lst_transforms.append(transfrm)

    # return as composition
    return transforms.Compose(lst_transforms)


class BaseTransforms(object):
    """Base class for implementing configurable transformations"""

    schema = BaseTransformsSchema

    configurables = None

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, sample):
        raise NotImplementedError("Please implement the `__call__` method")
