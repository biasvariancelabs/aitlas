import torch
from ..base import BaseMetric

import numpy as np


class F1score_segmentation(BaseMetric):
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

            predictions = predictions.to("cuda")
            labels = labels.to("cuda")

            tp = torch.sum(labels * predictions)
            fp = torch.sum(predictions) - tp
            fn = torch.sum(labels) - tp

            total_score += ((1 + beta ** 2) * tp + eps) \
                    / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

        return total_score / len(y_true)

