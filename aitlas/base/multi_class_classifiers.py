import logging
import os

import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim

from .models import BaseModel
from .schemas import BaseClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseMulticlassClassifier(BaseModel):
    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

    def get_predicted(self, outputs, threshold=None):
        probs = nnf.softmax(outputs.data, dim=1)
        predicted_probs, predicted = probs.topk(1, dim=1)
        return probs, predicted

    def log_additional_metrics(
        self,
        val_eval,
        y_true,
        y_pred,
        val_loss,
        dataset,
        model_directory,
        run_id,
        epoch,
    ):
        from ..visualizations import confusion_matrix

        fig = confusion_matrix(
            dataset,
            y_true,
            y_pred,
            os.path.join(model_directory, run_id, f"cm_{epoch + 1}.png"),
        )
        self.writer.add_figure("Confusion matrix", fig, epoch + 1)

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
