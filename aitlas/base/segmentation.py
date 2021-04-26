import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..metrics import F1ScoreSample
from ..utils import stringify
from .models import BaseModel
from .schemas import BaseSegmentationClassifierSchema
from .metrics import SegmentationRunningScore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseSegmentationClassifier(BaseModel):

    schema = BaseSegmentationClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = SegmentationRunningScore(self.num_classes, self.device)

    def get_predicted(self, outputs, threshold=None):
        predicted_probs = torch.tanh(outputs)
        predicted = (predicted_probs >= self.config.threshold).type(
            predicted_probs.dtype
        )
        return predicted_probs, predicted

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam([dict(params=self.model.parameters(), lr=self.config.learning_rate), ])

    def load_criterion(self):
        """Load the loss function"""
        return nn.MSELoss(reduction="mean")

    def load_lr_scheduler(self):
        return None

