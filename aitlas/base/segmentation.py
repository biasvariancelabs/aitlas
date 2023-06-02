import logging

import torch
import torch.optim as optim

from ..utils import DiceLoss
from .metrics import SegmentationRunningScore
from .models import BaseModel
from .schemas import BaseSegmentationClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseSegmentationClassifier(BaseModel):
    """Base class for a segmentation classifier.
    """

    schema = BaseSegmentationClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = SegmentationRunningScore(self.num_classes, self.device)

    def get_predicted(self, outputs, threshold=None):
        """Get predicted classes from the model outputs.

        :param outputs: Model outputs with shape (batch_size, num_classes).
        :type outputs: torch.Tensor
        :param threshold: The threshold for classification, defaults to None.
        :type threshold: float, optional
        :return: tuple containing the probabilities and predicted classes
        :rtype: tuple
        """
        predicted_probs = torch.sigmoid(outputs)
        predicted = (
            predicted_probs >= (threshold if threshold else self.config.threshold)
        ).long()
        return predicted_probs, predicted

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate)

    def load_criterion(self):
        """Load the loss function"""
        return DiceLoss()

    def load_lr_scheduler(self, optimizer):
        """Load the learning rate scheduler"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.1, min_lr=1e-6
        )
