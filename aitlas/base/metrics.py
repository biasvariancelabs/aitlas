import numpy as np


class BaseMetric:
    """Base class for implementing metrics """

    def __init__(self, device="cpu", **kwargs):
        self.device = device

    def calculate(self, y_true, y_pred):
        raise NotImplementedError("Please implement you metric calculation logic here.")


class RunningScore(object):
    """Generic metric container class. This class contains metrics that are averaged over batches. """

    def __init__(self, metrics, num_classes, device):
        self.calculated_metrics = {}
        self.metrics = metrics
        self.device = device
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            self.calculated_metrics[metric.name] = []

    def update(self, y_true, y_pred):
        """Updates stats on each batch"""

        # update metrics
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            calculated = metric.calculate(y_true, y_pred)
            if isinstance(calculated, dict):
                if isinstance(self.calculated_metrics[metric.name], list):
                    self.calculated_metrics[metric.name] = {}
                for k, v in calculated.items():
                    if not k in self.calculated_metrics[metric.name]:
                        self.calculated_metrics[metric.name][k] = []
                    self.calculated_metrics[metric.name][k].append(v)
            else:
                self.calculated_metrics[metric.name].append(calculated)

        # update confusion matrix
        for lt, lp in zip(y_true, y_pred):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.num_classes
            )

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int),
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            self.calculated_metrics[metric.name] = []

    def get_scores(self):
        metrics_summary = {}
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            if isinstance(self.calculated_metrics[metric.name], dict):
                metrics_summary[metric.name] = {}
                for k, v in self.calculated_metrics[metric.name].items():
                    metrics_summary[metric.name][k] = np.nanmean(
                        self.calculated_metrics[metric.name][k]
                    )
            else:
                metrics_summary[metric.name] = np.nanmean(
                    self.calculated_metrics[metric.name]
                )
        return metrics_summary

    def get_accuracy(self):
        hist = self.confusion_matrix
        accuracy = np.diag(hist).sum() / hist.sum()
        accuracy_per_class = np.diag(hist) / hist.sum(axis=1)
        accuracy_per_class = np.nanmean(accuracy_per_class)

        return {"accuracy": accuracy, "accuracy_per_class": accuracy_per_class}

    def get_iu(self):
        hist = self.confusion_matrix
        intersection_over_union = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
        mean_intersection_over_union = np.nanmean(intersection_over_union)
        intersection_over_union_per_class = dict(
            zip(range(self.num_classes), intersection_over_union)
        )

        return {
            "intersection_over_union": intersection_over_union,
            "intersection_over_union_per_class": intersection_over_union_per_class,
        }
