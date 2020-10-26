from .config import Configurable
from .schemas import BaseTransformsSchema


class BaseTransforms(Configurable):
    """Base class for implementing configurable transformations"""

    schema = BaseTransformsSchema

    def __init__(self, config=None):
        Configurable.__init__(self, config if config else {})

        self.transform = self.load_transforms()

    def __call__(self, input, target=None):
        raise NotImplementedError("Please implement the `__call__` method")

    def load_transforms(self):
        raise NotImplementedError("Please implement your transformations")
