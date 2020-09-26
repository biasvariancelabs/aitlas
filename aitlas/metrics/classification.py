from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..base import BaseMetric


class AccuracyScore(BaseMetric):
    name = "Accuracy"
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

        return {"Micro": micro, "Macro": macro, "Weighted": weighted}


class PrecisionScore(AveragedScore):
    name = "Precision"
    key = "precision"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = precision_score


class RecallScore(AveragedScore):
    name = "Recall"
    key = "recall"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = recall_score


class F1Score(AveragedScore):
    name = "F1 Score"
    key = "f1_score"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = f1_score
