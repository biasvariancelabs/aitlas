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
    """
    Base class for a multiclass classifier.

    Inherits from BaseModel.

    Attributes:
        schema (BaseClassifierSchema): The schema defines the classifier. See schema doc

    Methods:
        get_predicted(outputs, threshold=None): Get predicted classes from the model outputs.
        report(labels, dataset_name, running_metrics, **kwargs): Generate a report for multiclass classification.
        load_optimizer(): Load the optimizer for the classifier.
        load_criterion(): Load the loss function for the classifier.
        load_lr_scheduler(optimizer): Load the learning rate scheduler for the classifier.
    """


    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = MultiClassRunningScore(self.num_classes, self.device)

    def get_predicted(self, outputs, threshold=None):
        """
        Get predicted classes from the model outputs.

        Args:
            outputs (torch.Tensor): Model outputs with shape (batch_size, num_classes).
            threshold (float, optional): Threshold for classification. Defaults to None.

        Returns:
            tuple: Tuple containing the probabilities and predicted classes.
        """

        probs = nnf.softmax(outputs.data, dim=1)
        predicted_probs, predicted = probs.topk(1, dim=1)
        return probs, predicted

    def report(self, labels, dataset_name, running_metrics, **kwargs):
        """
        Generate a report for multiclass classification.

        Args:

            :param labels (list): List of class labels.
            dataset_name (str): Name of the dataset.
            running_metrics (RunningScore): A running score object for multiclass classification.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        run_id = kwargs.get("id", "experiment")
        from ..visualizations import plot_multiclass_confusion_matrix

        if running_metrics.confusion_matrix:
            cm = running_metrics.get_computed()

        # plot confusion matrix for model evaluation
        plot_multiclass_confusion_matrix(
            np.array(cm), labels, dataset_name, f"{dataset_name}_{self.name}_{run_id}_cm.pdf"
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
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, min_lr=1e-6)


class BaseMultilabelClassifier(BaseModel):
    """
    Base class for a multilabel classifier.

    Inherits from BaseModel.

    Attributes:
        schema (BaseClassifierSchema): The schema for the classifier.

    Methods:
        get_predicted(outputs, threshold=None): Get predicted classes from the model outputs.
        report(labels, dataset_name, running_metrics, **kwargs): Generate a report for multilabel classification.
        load_optimizer(): Load the optimizer for the classifier.
        load_criterion(): Load the loss function for the classifier.
        load_lr_scheduler(optimizer): Load the learning rate scheduler for the classifier.
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
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, min_lr=1e-6)

    def get_predicted(self, outputs, threshold=None):
        """
        Get predicted classes from the model outputs.

        Args:
            outputs (torch.Tensor): Model outputs with shape (batch_size, num_classes).
            threshold (float, optional): Threshold for classification. Defaults to None.

        Returns:
        
            tuple: Tuple containing the probabilities and predicted classes.
        """
        predicted_probs = torch.sigmoid(outputs)
        predicted = predicted_probs >= (
            threshold if threshold else self.config.threshold
        )
        return predicted_probs, predicted

    def report(self, labels, dataset_name, running_metrics, **kwargs):
        """ Generate a report for multilabel classification.
      
        :param labels: List of class labels
        :param dataset_name: Name of the dataset
        :param running_metrics: type of metrics to be reported. Currently only confusion matrix is supported"""
        
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
            np.array(cm_array), labels, dataset_name, f"{dataset_name}_{self.name}_{run_id}_cm.pdf"
        )
