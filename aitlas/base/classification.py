import logging

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim

from .models import BaseModel
from .schemas import BaseClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseMulticlassClassifier(BaseModel):
    """Base class for a multiclass classifier"""

    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

    def get_predicted(self, outputs, threshold=None):
        probs = nnf.softmax(outputs.data, dim=1)
        predicted_probs, predicted = probs.topk(1, dim=1)
        return probs, predicted

    def report(self, y_true, y_pred, y_prob, labels, **kwargs):
        """Report for multiclass classification"""
        run_id = kwargs.get("id", "experiment")
        from ..visualizations import confusion_matrix, precision_recall_curve

        # plot confusion matrix for model evaluation
        confusion_matrix(y_true, y_pred, y_prob, labels, f"{run_id}_cm.png")

        # plot roc curve for model evaluation
        precision_recall_curve(y_true, y_pred, y_prob, labels, f"{run_id}_pr.png")

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.SGD(
            self.model.parameters(), lr=self.config.learning_rate, momentum=0.9
        )

    def load_criterion(self):
        """Load the loss function"""
        return nn.CrossEntropyLoss()

    def load_lr_scheduler(self):
        return None


class BaseMultilabelClassifier(BaseModel):
    """Base class for a multilabel classifier"""

    schema = BaseClassifierSchema

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.SGD(
            self.model.parameters(), lr=self.config.learning_rate, momentum=0.9
        )

    def load_criterion(self):
        """Load the loss function"""
        return nn.CrossEntropyLoss()

    def load_lr_scheduler(self):
        return None

    def get_predicted(self, outputs, threshold=None):
        predicted_probs = torch.sigmoid(outputs)
        predicted = (predicted_probs >= self.config.threshold).type(
            predicted_probs.dtype
        )
        return predicted_probs, predicted
