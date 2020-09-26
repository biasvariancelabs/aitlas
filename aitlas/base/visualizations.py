from .config import Configurable


class BaseVisualization:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self):
        raise NotImplementedError
