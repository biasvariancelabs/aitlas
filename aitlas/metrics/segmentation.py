import torch
from ..base import BaseMetric


class F1score_segmentation(BaseMetric):
    name = "F1 Score"
    key = "f1_score"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred, beta=1, eps=1e-7):
        y_true = y_true[0]
        y_pred = y_pred[0]
        tp = torch.sum(y_true * y_pred)
        fp = torch.sum(y_pred) - tp
        fn = torch.sum(y_true) - tp

        score = ((1 + beta ** 2) * tp + eps) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

        return {"F1 Score": score}

