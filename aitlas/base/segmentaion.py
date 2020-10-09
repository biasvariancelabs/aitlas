import torch

from .classifiers import BaseClassifier
from .schemas import BaseClassifierSchema


class BaseSegmentation(BaseClassifier):
    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

    def load_lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(
            self.load_optimizer(), step_size=3, gamma=0.1
        )
