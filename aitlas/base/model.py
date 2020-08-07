import torch.nn as nn

from .config import Configurable


class BaseModel(nn.Module, Configurable):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        Configurable.__init__(self, config)

    def train(self, epochs: int = 10):
        """
        Trains the model on the given dataset. Saves the model on disk for reuse.
        """
        pass

    def predict(self):
        """
        Makes predictions for a given model and dataset.
        """
        pass

    def evaluate(self):
        """
        Evaluates a given model against a test dataset.
        """
        pass

    def forward(self, *input, **kwargs):
        """
        Abstract method implementing the model. Extending classes should override this method.
        :return:  instance extending `nn.Module`
        """
        raise NotImplementedError
