from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


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


class PrecisionScore(BaseMetric):
    name = "Precision"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def calculate(self, y_true, y_pred):
        micro = precision_score(y_true, y_pred, average="micro")
        macro = precision_score(y_true, y_pred, average="macro")
        weighted = precision_score(y_true, y_pred, average="weighted")

        return {"micro": micro, "macro": macro, "weighted": weighted}


class RecallScore(BaseMetric):
    name = "Recall"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def calculate(self, y_true, y_pred):
        micro = recall_score(y_true, y_pred, average="micro")
        macro = recall_score(y_true, y_pred, average="macro")
        weighted = recall_score(y_true, y_pred, average="weighted")

        return {"micro": micro, "macro": macro, "weighted": weighted}


class F1Score(BaseMetric):
    name = "F1 Score"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def calculate(self, y_true, y_pred):
        micro = f1_score(y_true, y_pred, average="micro")
        macro = f1_score(y_true, y_pred, average="macro")
        weighted = f1_score(y_true, y_pred, average="weighted")

        return {"micro": micro, "macro": macro, "weighted": weighted}


# Available metrics. Add keys with new metrics here.
CLASSIFICATION_METRICS = {
    "accuracy": AccuracyScore,
    "precision": PrecisionScore,
    "recall": RecallScore,
    "f1_score": F1Score,
}
