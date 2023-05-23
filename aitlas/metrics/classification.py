"""Metrics for classification tasks."""


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..base import BaseMetric


class AccuracyScore(BaseMetric):
    """
    Accuracy score class, inherits from BaseMetric.
    """

    name = "accuracy"
    key = "accuracy"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def calculate(self, y_true, y_pred):
        """
        Computes the Accuracy score.

        Given model predictions for a target variable, it calculates the
        accuracy score as the number of correct predictionsdivided
        by the total number of predictions.

        :parm y_true: The ground truth values for the target variable.
        :type y_true: array-like of arbitrary size
        :parm y_pred: The prediction values for the target variable.
        :type y_pred: array-like of identical size as y_true
        :return: A number in [0, 1] where, 1 is a perfect classification.
        :rtype: float
        """
        return accuracy_score(y_true, y_pred)


class AveragedScore(BaseMetric):
    """
    Average score class. Inherits from BaseMetric.
    """

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred):
        """ "
        It calculates the score for each class and then averages the results.
        The type of average is {'micro', 'macro', 'weighted'}:
        *'micro': Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        *'macro': Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        *'weighted': Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance.

        :param y_true: The ground truth labels
        :type y_true: array-like
        :param y_pred: The predicted labels
        :type y_pred: array-like
        :return: A dictionary with the micro, macro and weighted average scores
        :rtype: dict
        :raises ValueError: If the shapes of y_pred and y_true do not match.

        """

        micro = self.method(y_true, y_pred, average="micro")
        macro = self.method(y_true, y_pred, average="macro")
        weighted = self.method(y_true, y_pred, average="weighted")

        return {"micro": micro, "macro": macro, "weighted": weighted}


class PrecisionScore(AveragedScore):
    """Precision score class, inherits from AveragedScore."""

    name = "precision"
    key = "precision"

    def __init__(self, **kwargs):
        AveragedScore.__init__(self, **kwargs)
        self.method = precision_score


class RecallScore(AveragedScore):
    """Precision score class, inherits from AveragedScore."""

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
