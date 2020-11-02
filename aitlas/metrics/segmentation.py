import numpy as np
import torch

from ..base import BaseMetric


class F1ScoreSample(BaseMetric):
    name = "F1 Score"
    key = "f1_score"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred, beta=1, eps=1e-7):
        total_score = 0.0
        for i, item in enumerate(y_true):
            predictions = torch.from_numpy(np.array(y_pred[i]))
            labels = torch.from_numpy(np.array(y_true[i]))

            predictions = predictions.to(self.device)
            labels = labels.to(self.device)

            tp = torch.sum(labels * predictions)
            fp = torch.sum(predictions) - tp
            fn = torch.sum(labels) - tp

            total_score += ((1 + beta ** 2) * tp + eps) / (
                (1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps
            )

        return float(total_score / len(y_true))


class IoU(BaseMetric):
    name = "IoU"
    key = "iou"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred, eps=1e-7):
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
    name = "Accuracy"
    key = "accuracy"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred):
        total_score = 0.0
        for i, item in enumerate(y_true):
            predictions = torch.from_numpy(np.array(y_pred[i]))
            labels = torch.from_numpy(np.array(y_true[i]))

            predictions = predictions.to(self.device)
            labels = labels.to(self.device)

            tp = torch.sum(labels == predictions, dtype=predictions.dtype)
            total_score += tp / labels.view(-1).shape[0]

        return float(total_score / len(y_true))
