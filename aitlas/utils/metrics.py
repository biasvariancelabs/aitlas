from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class BaseMetric:
    """Base class for implementing metrics """

    name = "BaseMetric"  # override this to give your metric a descriptive name

    def __init__(self, **kwargs):
        pass

    def calculate(self, y_true, y_pred):
        raise NotImplementedError


class AccuracyScore(BaseMetric):
    name = "Accuracy"

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

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = precision_score


class RecallScore(AveragedScore):
    name = "Recall"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = recall_score


class F1Score(AveragedScore):
    name = "F1 Score"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = f1_score


# Available metrics. Add keys with new metrics here.
CLASSIFICATION_METRICS = {
    "accuracy": AccuracyScore,
    "precision": PrecisionScore,
    "recall": RecallScore,
    "f1_score": F1Score,
}
