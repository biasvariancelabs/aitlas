import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ..metrics import F1score_segmentation, F1Score
from .models import BaseModel
from .schemas import BaseClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseSegmentationClassifier(BaseModel):
    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

    def get_predicted(self, outputs, threshold=None):
        predicted_probs = torch.tanh(outputs)
        predicted = (predicted_probs >= 0.3).type(predicted_probs.dtype)
        return predicted_probs, predicted

    def metrics(self):
        return [F1score_segmentation]

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam([
        dict(params=self.model.parameters(), lr=0.0001),
    ])

    def load_criterion(self):
        """Load the loss function"""
        return nn.MSELoss(reduction='mean')

    def load_lr_scheduler(self):
        return None

    #def load_lr_scheduler(self):
    #    return torch.optim.lr_scheduler.StepLR(
    #        self.load_optimizer(), step_size=3, gamma=0.1
    #    )