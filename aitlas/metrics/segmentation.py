"""Metrics for segmentation tasks."""
import numpy as np
import torch

from ..base import BaseMetric


class F1ScoreSample(BaseMetric):
    """
    Calculates the F1 score metric for binary segmentation tasks.

    """

    name = "F1 Score"
    key = "f1_score"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred, beta=1, eps=1e-7):
        """
        Calculate the F1 Score.

        :param y_true: True labels
        :type y_true: list or numpy array
        :param y_pred: Predicted labels
        :type y_pred: list or numpy array
        :param beta: Weight of precision in the combined score. Default is 1.
        :type beta: float
        :param eps: Small value to prevent zero division. Default is 1e-7.
        :type eps: float
        :return: F1 score
        :rtype: float
        :raises ValueError: If the shapes of y_pred and y_true do not match.
        """
        total_score = 0.0
        for i, item in enumerate(y_true):
            predictions = torch.from_numpy(np.array(y_pred[i]))
            labels = torch.from_numpy(np.array(y_true[i]))

            predictions = predictions.to(self.device)
            labels = labels.to(self.device)

            tp = torch.sum(labels * predictions)
            fp = torch.sum(predictions) - tp
            fn = torch.sum(labels) - tp

            total_score += ((1 + beta**2) * tp + eps) / (
                (1 + beta**2) * tp + beta**2 * fn + fp + eps
            )

        return float(total_score / len(y_true))


class IoU(BaseMetric):
    """
    Calculates the Intersection over Union (IoU) metric for binary segmentation tasks.
    """

    name = "IoU"
    key = "iou"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred, eps=1e-7):
        """
        Calculate the IoU score.

        :param y_true: True labels
        :type y_true: list or numpy array
        :param y_pred: Predicted labels
        :type y_pred: list or numpy array
        :param eps: Small value to prevent zero division. Default is 1e-7.
        :type eps: float
        :return: IoU score
        :rtype: float
        :raises ValueError: If the shapes of y_pred and y_true do not match.
        """

        total_score = 0.0
        for i, item in enumerate(y_true):
            predictions = torch.from_numpy(np.array(y_pred[i]))
            labels = torch.from_numpy(np.array(y_true[i]))

            predictions = predictions.to(self.device)
            labels = labels.to(self.device)

            intersection = torch.sum(labels * predictions)
            union = torch.sum(labels) + torch.sum(predictions) - intersection + eps
            total_score += (intersection + eps) / union

        return float(total_score / len(y_true))


class Accuracy(BaseMetric):
    """
    Calculates the accuracy metric.
    """

    name = "Accuracy"
    key = "accuracy"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred):
        """
        Calculate accuracy.

        :param y_true: True labels
        :type y_true: list or numpy array
        :param y_pred: Predicted labels
        :type y_pred: list or numpy array
        :return: Accuracy score
        :rtype: float
        """

        total_score = 0.0
        for i, item in enumerate(y_true):
            predictions = torch.from_numpy(np.array(y_pred[i]))
            labels = torch.from_numpy(np.array(y_true[i]))

            predictions = predictions.to(self.device)
            labels = labels.to(self.device)

            tp = torch.sum(labels == predictions, dtype=predictions.dtype)
            total_score += tp / labels.view(-1).shape[0]

        return float(total_score / len(y_true))


class DiceCoefficient(BaseMetric):
    """
    A Dice Coefficient metic, used to evaluate the similarity of two sets.

    .. note:: More information on its Wikipedia page: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    """

    name = "DiceCoefficient"
    key = "dice_coefficient"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def calculate(self, y_true, y_pred):
        """
        Method to compute the Dice coefficient.

        Given two sets X and Y, the coefficient is calculated as:
        .. math::
             DSC = {2 * | X intersection Y |} / {|X| + |Y|},  where |X| and |Y| are the cardinalities of the two sets.

        .. note:: Based on the implementation at: https://github.com/CosmiQ/cresi/blob/master/cresi/net/pytorch_utils/loss.py#L47

        :param y_true: The ground truth values for the target variable. Can be array-like of arbitrary size.
        :type y_true: list or numpy array
        :param y_pred: The prediction values for the target variable. Must be of identical size as y_true.
        :type y_pred: list or numpy array
        :return: A number in [0, 1] where 0 equals no similarity and 1 is maximum similarity.
        :rtype: float
        :raises ValueError: If the shapes of y_pred and y_true do not match.
        """
        # If the parameters are passed as lists, convert them to tensors
        if isinstance(y_true, list):
            y_true = torch.from_numpy(np.array(y_true))
        if isinstance(y_pred, list):
            y_pred = torch.from_numpy(np.array(y_pred))
        # Check shape equality
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch, y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape"
            )
        batch_size = len(y_true)
        # Flatten images (N, C, H, W) => (N, C*H*W)
        predictions = y_pred.view(batch_size, -1)
        labels = y_true.view(batch_size, -1)
        # Calculate intersection and numerator values
        intersection = (predictions * labels).sum(1)
        numerator = predictions.sum(1) + labels.sum(1)
        # Calculate final scores
        scores = (2.0 * intersection) / numerator
        # Average over the batch
        score = scores.sum() / batch_size
        return torch.clamp(score, 0.0, 1.0)


class FocalLoss(BaseMetric):
    """
    Class for calculating Focal Loss, a loss metric that extends the binary cross entropy loss. Focal loss reduces the relative loss for well-classified examples and puts more focus on hard, misclassified examples.
    Computed as:
    .. math:: alpha * (1-bce_loss)**gamma

    .. note:: For more information, refer to the papers: https://paperswithcode.com/method/focal-loss, and: https://amaarora.github.io/2020/06/29/FocalLoss.html
    """

    name = "FocalLoss"
    key = "focal_loss"

    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True, **kwargs):
        """
        Intilisation.

        :param alpha: Weight parameter. Default is 1.
        :type alpha: int
        :param gamma: Focusing parameter. Default is 2.
        :type gamma: int
        :param logits: Controls whether probabilities or raw logits are passed. Default is True.
        :type logits: bool
        :param reduce: Specifies whether to reduce the loss to a single value. Default is True.
        :type reduce: bool
        :param kwargs: Any key word arguments to be passed to the base class
        """
        BaseMetric.__init__(self, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def calculate(self, y_true, y_pred):
        """
         Method to compute the focal loss.

         .. note:: Based on the implementation at: https://www.kaggle.com/c/tgs

        :param y_true: The ground truth values for the target variable. Can be array-like of arbitrary size.
         :type y_true: list or numpy array
         :param y_pred: The prediction values for the target variable. Must be of identical size as y_true.
         :type y_pred: list or numpy array
         :return: The focal loss between y_pred and y_true.
         :rtype: float
         :raises ValueError: If the shapes of y_pred and y_true do not match.
        """
        # If the parameters are passed as lists, convert them to tensors
        if isinstance(y_true, list):
            y_true = torch.from_numpy(np.array(y_true))
        if isinstance(y_pred, list):
            y_pred = torch.from_numpy(np.array(y_pred))
        # Check shape equality
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"shape mismatch, y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape"
            )

        def loss(x, y):
            """The actual FocalLoss implementation."""
            import torch.nn.functional as F

            if self.logits:
                binary_cross_entropy_loss = F.binary_cross_entropy_with_logits(
                    input=x, target=y
                )
            else:
                binary_cross_entropy_loss = F.binary_cross_entropy(input=x, target=y)
            pt = torch.exp(-1 * binary_cross_entropy_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * binary_cross_entropy_loss
            if self.reduce:
                return torch.mean(focal_loss)
            else:
                return focal_loss

        batch_size = len(y_true)
        score = 0.0
        # Iterates through each item in the batch
        for inx, _ in enumerate(y_true):
            score += loss(y_pred[inx], y_true[inx])
        return score / batch_size


class CompositeMetric(BaseMetric):
    """
    A class for combining multiple metrics.
    """

    name = "CompositeMetric"
    key = "composite_metric"

    def __init__(self, metrics=None, weights=None, **kwargs):
        """
        Initialisation.

        :param metrics: A list of metrics that subclass the BaseMetric class and have valid implementation of calculate(y_true, y_pred). Default is None.
        :type metrics: list
        :param weights: A list of floats who sum up to 1. Default is None.
        :type weights: list
        :param kwargs: Any key word arguments to be passed to the base class
        :raises ValueError: If the length of metrics and weights is not equal or if the sum of weights is not equal to one.
        """
        BaseMetric.__init__(self, **kwargs)
        if len(metrics) != len(weights):
            raise ValueError(
                f"the lists of metrics ({len(metrics)}) and weights ({len(weights)}) must be of equal length"
            )
        if sum(weights) != 1:
            raise ValueError(
                f"the sum of weights ({sum(weights)}) must be equal to one"
            )
        self.zipped = zip(weights, metrics)

    def calculate(self, y_true, y_pred):
        """
        Method to calculate the weighted sum of the metric values.

        :param y_true: The ground truth values for the target variable. Can be array-like of arbitrary size.
        :type y_true: list or numpy array
        :param y_pred: The prediction values for the target variable. Must be of identical size as y_true.
        :type y_pred: list or numpy array
        :return: The weighted sum of each metric value.
        :rtype: float
        :raises ValueError: If the shapes of y_pred and y_true do not match.
        """
        # If the parameters are passed as lists, convert them to tensors
        if isinstance(y_true, list):
            y_true = torch.from_numpy(np.array(y_true))
        if isinstance(y_pred, list):
            y_pred = torch.from_numpy(np.array(y_pred))
        # Check shape equality
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch, y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape"
            )
        from itertools import starmap

        def calculate_weighted(weight, metric):
            return metric.calculate(y_true=y_true, y_pred=y_pred) * weight

        return sum(starmap(calculate_weighted, self.zipped))
