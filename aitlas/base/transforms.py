from .config import Configurable
from .schemas import BaseTranformsSchema


class BaseTransforms(Configurable):
    """Base class for implementing configurable transformations"""

    schema = BaseTranformsSchema

    def __init__(self, config):
        Configurable.__init__(self, config)

    def __call__(self, input, target):
        raise NotImplementedError("Please implement the `__call__` method")
