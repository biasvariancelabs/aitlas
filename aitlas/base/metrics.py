import numpy as np


class BaseMetric:
    """Base class for implementing metrics """

    def __init__(self, device="cpu", **kwargs):
        self.device = device

    def calculate(self, y_true, y_pred):
        raise NotImplementedError("Please implement you metric calculation logic here.")


class runningScore1(object):
    def __init__(self, metrics, device):
        self.calculated_metrics = {}
        self.metrics = metrics
        self.device = device
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            self.calculated_metrics[metric.name] = []

    def update(self, y_true, y_pred):
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            self.calculated_metrics[metric.name].append(metric.calculate(y_true, y_pred))

    def reset(self):
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            self.calculated_metrics[metric.name] = []

    def get_scores(self):
        metrics_summary = {}
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            metrics_summary[metric.name] = np.nanmean(self.calculated_metrics[metric.name])
        return metrics_summary

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))