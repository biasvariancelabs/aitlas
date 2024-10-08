import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim

from .metrics import MultiClassRunningScore, MultiLabelRunningScore
from .models import BaseModel
from .schemas import BaseClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseMulticlassClassifier(BaseModel):
    """Base class for a multiclass classifier.
    """

    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = MultiClassRunningScore(self.num_classes, self.device)

    def get_predicted(self, outputs, threshold=None):
        """Get predicted classes from the model outputs.

        :param outputs: Model outputs with shape (batch_size, num_classes).
        :type outputs: torch.Tensor
        :param threshold: The threshold for classification, defaults to None.
        :type threshold: float, optional
        :return: tuple containing the probabilities and predicted classes
        :rtype: tuple
        """
        probs = nnf.softmax(outputs.data, dim=1)
        predicted_probs, predicted = probs.topk(1, dim=1)
        return probs, predicted

    def report(self, labels, dataset_name, running_metrics, **kwargs):
        """Generate a report for multiclass classification.

        :param labels: List of class labels.
        :type labels: list
        :param dataset_name: Name of the dataset.
        :type dataset_name: list
        :param running_metrics: A running score object for multiclass classification.
        :type running_metrics: aitlas.base.metrics.RunningScore
        """
        run_id = kwargs.get("id", "experiment")
        from ..visualizations import plot_multiclass_confusion_matrix

        if running_metrics.confusion_matrix:
            cm = running_metrics.get_computed()

        # plot confusion matrix for model evaluation
        plot_multiclass_confusion_matrix(
            np.array(cm),
            labels,
            dataset_name,
            f"{dataset_name}_{self.name}_{run_id}_cm.pdf",
        )

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.RAdam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def load_criterion(self):
        """Load the loss function"""
        return nn.CrossEntropyLoss(weight=self.weights)

    def load_lr_scheduler(self, optimizer):
        """Load the learning rate scheduler"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.1, min_lr=1e-6
        )


class BaseMultilabelClassifier(BaseModel):
    """Base class for a multilabel classifier.
    """

    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = MultiLabelRunningScore(self.num_classes, self.device)

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.RAdam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def load_criterion(self):
        """Load the loss function"""
        return nn.BCEWithLogitsLoss(weight=self.weights)

    def load_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.1, min_lr=1e-6
        )

    def get_predicted(self, outputs, threshold=None):
        """Get predicted classes from the model outputs.

        :param outputs: Model outputs with shape (batch_size, num_classes).
        :type outputs: torch.Tensor
        :param threshold: Threshold for classification, defaults to None
        :type threshold: float, optional
        :return: Tuple containing the probabilities and predicted classes.
        :rtype: tuple
        """
        predicted_probs = torch.sigmoid(outputs)
        predicted = predicted_probs >= (
            threshold if threshold else self.config.threshold
        )
        return predicted_probs, predicted

    def report(self, labels, dataset_name, running_metrics, **kwargs):
        """Generate a report for multilabel classification.

        :param labels: List of class labels
        :type labels: list
        :param dataset_name: Name of the dataset.
        :type dataset_name: str
        :param running_metrics: Type of metrics to be reported. Currently only confusion matrix is
        :type running_metrics: aitlas.base.metrics.RunningScore
        """

        run_id = kwargs.get("id", "experiment")
        cm_array = []
        if running_metrics.confusion_matrix:
            cm = running_metrics.get_computed()
            for i, label in enumerate(labels):
                tp = cm[i, 1, 1]
                tn = cm[i, 0, 0]
                fp = cm[i, 0, 1]
                fn = cm[i, 1, 0]
                cm_array.append([[int(tn), int(fp)], [int(fn), int(tp)]])

        from ..visualizations import plot_multilabel_confusion_matrix

        # plot confusion matrix for model evaluation
        plot_multilabel_confusion_matrix(
            np.array(cm_array),
            labels,
            dataset_name,
            f"{dataset_name}_{self.name}_{run_id}_cm.pdf",
        )
