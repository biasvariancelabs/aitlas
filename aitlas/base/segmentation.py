import torch

from .multi_class_classifiers import BaseMulticlassClassifier
from .schemas import BaseClassifierSchema


class BaseSegmentation(BaseMulticlassClassifier):
    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

    def load_lr_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(
            self.load_optimizer(), step_size=3, gamma=0.1
        )
