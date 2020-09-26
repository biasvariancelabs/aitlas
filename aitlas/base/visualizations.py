from .config import Configurable


class BaseVisualization:
    def __init__(self, **kwargs):
        pass

    def plot(self):
        raise NotImplementedError
