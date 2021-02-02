from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..base import BaseMetric


class AccuracyScore(BaseMetric):
    name = "accuracy"
    key = "accuracy"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def calculate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


class AveragedScore(BaseMetric):
    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred):
        micro = self.method(y_true, y_pred, average="micro")
        macro = self.method(y_true, y_pred, average="macro")
        weighted = self.method(y_true, y_pred, average="weighted")

        return {"micro": micro, "macro": macro, "weighted": weighted}


class PrecisionScore(AveragedScore):
    name = "precision"
    key = "precision"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = precision_score


class RecallScore(AveragedScore):
    name = "recall"
    key = "recall"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = recall_score


class F1Score(AveragedScore):
    name = "f1 score"
    key = "f1_score"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = f1_score
