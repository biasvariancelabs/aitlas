import numpy as np
from ignite.metrics.multilabel_confusion_matrix import MultiLabelConfusionMatrix

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
        self.confusion_matrix = MultiLabelConfusionMatrix(num_classes=self.num_classes, device=self.device)
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            self.calculated_metrics[metric.name] = []

    def update(self, y_true, y_pred):
        """Updates stats on each batch"""
        self.confusion_matrix.update((y_pred, y_true))

    def reset(self):
        self.confusion_matrix.reset()
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

    def get_f1score(self):
        cm = self.confusion_matrix.compute().cpu().detach().numpy()
        tn_overall = np.sum(cm[:, 0, 0])
        tp_overall = np.sum(cm[:, 1, 1])
        fn_overall = np.sum(cm[:, 1, 0])
        fp_overall = np.sum(cm[:, 0, 1])
        precision_overall = tp_overall / (tp_overall + fp_overall)
        recall_overall = tp_overall / (tp_overall + fn_overall)
        micro_f1score = (2 * precision_overall * recall_overall) / (precision_overall + recall_overall)

        macro_f1score = []
        weights = []
        total_samples = np.sum(cm[:, 0, 1]) + np.sum(cm[:, 1, 1])
        f1score_per_class = []
        for i in range(self.num_classes):
            tn = cm[i, 0, 0]
            tp = cm[i, 1, 1]
            fn = cm[i, 1, 0]
            fp = cm[i, 0, 1]
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            weights.append((tp + fp) / total_samples)
            macro_f1score.append((2 * precision * recall) / (precision + recall))
            f1score_per_class.append((2 * precision * recall) / (precision + recall))
        macro_f1score = np.array(macro_f1score)
        macro_f1score[np.isnan(macro_f1score)] = 0
        # calculate weighted F1 score
        weighted_f1score = macro_f1score * weights

        f1score_per_class = np.array(f1score_per_class)
        f1score_per_class[np.isnan(f1score_per_class)] = 0

        return {"Micro F1-score": micro_f1score,
                "Macro F1-score": np.mean(macro_f1score),
                "Weighted F1-score": np.sum(weighted_f1score),
                "F1-score per class": np.array(f1score_per_class)}

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
