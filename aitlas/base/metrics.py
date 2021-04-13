import numpy as np
import torch
from ignite.metrics import confusion_matrix
from ignite.metrics.multilabel_confusion_matrix import MultiLabelConfusionMatrix


class BaseMetric:
    """Base class for implementing metrics """

    def __init__(self, device="cpu", **kwargs):
        self.device = device

    def calculate(self, y_true, y_pred):
        raise NotImplementedError("Please implement you metric calculation logic here.")


class RunningScore(object):
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.confusion_matrix = None

    def update(self, y_true, y_pred):
        """Updates stats on each batch"""
        self.confusion_matrix.update((y_pred, y_true))

    def reset(self):
        """Reset the confusion matrix"""
        self.confusion_matrix.reset()

    def get_computed(self):
        return self.confusion_matrix.compute().type(torch.DoubleTensor)

    def precision(self):
        raise NotImplementedError

    def accuracy(self):
        raise NotImplementedError

    def recall(self):
        raise NotImplementedError

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        micro = (
            2
            * precision["Precision Micro"]
            * recall["Recall Micro"]
            / (precision["Precision Micro"] + recall["Recall Micro"] + 1e-15)
        )
        macro = (
            2
            * precision["Precision Macro"]
            * recall["Recall Macro"]
            / (precision["Precision Macro"] + recall["Recall Macro"] + 1e-15)
        )
        weighted = (
            2
            * precision["Precision Weighted"]
            * recall["Recall Weighted"]
            / (precision["Precision Weighted"] + recall["Recall Weighted"] + 1e-15)
        )
        per_class = (
            2
            * precision["Precision per Class"]
            * recall["Recall per Class"]
            / (precision["Precision per Class"] + recall["Recall per Class"] + 1e-15)
        )

        return {
            "F1_score Micro": float(micro),
            "F1_score Macro": float(macro),
            "F1_score Weighted": float(weighted),
            "F1_score per Class": per_class.tolist(),
        }

    def iou(self):
        raise NotImplementedError

    def get_scores(self, metrics):
        """Returns the specified metrics"""
        result = []
        for metric in metrics:
            result.append(getattr(self, metric)())
        return result


class MultiClassRunningScore(RunningScore):
    """Calculates confusion matrix for multi-class data. This class contains metrics that are averaged over batches. """

    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        self.confusion_matrix = confusion_matrix.ConfusionMatrix(
            num_classes=num_classes, device=device
        )

    def accuracy(self):
        cm = self.get_computed()
        accuracy = cm.diag().sum() / (cm.sum() + 1e-15)
        return {"Accuracy": accuracy}

    def recall(self):
        cm = self.get_computed()
        micro = cm.diag().sum() / (cm.sum() + 1e-15)  # same as accuracy for multiclass
        macro = (cm.diag() / (cm.sum(dim=1) + 1e-15)).mean()
        weighted = (
            (cm.diag() / (cm.sum(dim=1) + 1e-15))
            * ((cm.sum(dim=1)) / (cm.sum() + 1e-15))
        ).sum()
        per_class = cm.diag() / (cm.sum(dim=1) + 1e-15)

        return {
            "Recall Micro": micro,
            "Recall Macro": macro,
            "Recall Weighted": weighted,
            "Recall per Class": per_class,
        }

    def precision(self):
        cm = self.get_computed()
        micro = cm.diag().sum() / (cm.sum() + 1e-15)  # same as accuracy for multiclass
        macro = (cm.diag() / (cm.sum(dim=0) + 1e-15)).mean()
        weighted = (
            (cm.diag() / (cm.sum(dim=0) + 1e-15))
            * ((cm.sum(dim=1)) / (cm.sum() + 1e-15))
        ).sum()
        per_class = cm.diag() / (cm.sum(dim=0) + 1e-15)

        return {
            "Precision Micro": micro,
            "Precision Macro": macro,
            "Precision Weighted": weighted,
            "Precision per Class": per_class,
        }

    def iou(self):
        cm = self.get_computed()
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)

        return {"IOU": iou.tolist(), "mIOU": float(iou.mean())}


class MultiLabelRunningScore(RunningScore):
    """Calculates a confusion matrix for multi-labelled, multi-class data in addition to the """

    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        self.confusion_matrix = MultiLabelConfusionMatrix(
            num_classes=self.num_classes, device=self.device
        )

    def accuracy(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)

        accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)
        accuracy_per_class = (tp + tn) / (tp + tn + fp + fn)

        return {"Accuracy": accuracy, "Accuracy per Class": accuracy_per_class}

    def precision(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)

        micro = tp_total / (tp_total + fp_total)
        per_class = tp / (tp + fp)
        macro = per_class.mean()
        weighted = (per_class * (tp + fn) / self.count()).sum()

        return {
            "Precision Micro": micro,
            "Precision Macro": macro,
            "Precision Weighted": weighted,
            "Precision per Class": per_class,
        }

    def recall(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)

        micro = tp_total / (tp_total + fn_total)
        per_class = tp / (tp + fn)
        macro = per_class.mean()
        weighted = (per_class * (tp + fn) / self.count()).sum()

        return {
            "Recall Micro": micro,
            "Recall Macro": macro,
            "Recall Weighted": weighted,
            "Recall per Class": per_class,
        }

    def get_outcomes(self, total=False):
        """
        Return true/false positives/negatives from the confusion matrix
        :param total: do we need to return per class or total
        """
        cm = self.get_computed()

        tp = cm[:, 1, 1]
        tn = cm[:, 0, 0]
        fp = cm[:, 0, 1]
        fn = cm[:, 1, 0]

        if total:  # sum it all if we need to calculate to totals
            tp, tn, fp, fn = tp.sum(), tn.sum(), fp.sum(), fn.sum()

        return tp, tn, fp, fn

    def count(self):
        tp, tn, fp, fn = self.get_outcomes(True)
        return tp + tn + fp + fn

    def get_samples(self):
        cm = self.confusion_matrix.compute().cpu().detach().numpy()
        return np.sum(cm[:, 0, 1]) + np.sum(cm[:, 1, 1])

    def iou(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)

        iou_per_class = tp / (tp + fp + fn)
        iou = tp_total / (tp_total + fp_total + fn_total)

        return {
            "IOU": iou,
            "IOU mean": iou_per_class.mean(),
            "IOU pre Class": iou_per_class,
        }

    #
    # tp_overall, tn_overall, fp_overall, fn_overall = self.get_outcomes()
    # print(f"{tp_overall.shape}, {tn_overall.shape}, {fp_overall.shape}, {fn_overall.shape}")
    # precision_overall = tp_overall / (tp_overall + fp_overall)
    # recall_overall = tp_overall / (tp_overall + fn_overall)
    # micro_f1score = (2 * precision_overall * recall_overall) / (precision_overall + recall_overall)
    #
    # macro_f1score = []
    # weights = []
    # f1score_per_class = []
    # for i in range(self.num_classes):
    #     tp, tf, fp, fn = self.get_outcomes(i)
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     weights.append((tp + fp) / self.get_samples())
    #     macro_f1score.append((2 * precision * recall) / (precision + recall))
    #     f1score_per_class.append((2 * precision * recall) / (precision + recall))
    # macro_f1score = np.array(macro_f1score)
    # macro_f1score[np.isnan(macro_f1score)] = 0
    # # calculate weighted F1 score
    # weighted_f1score = macro_f1score * weights
    #
    # f1score_per_class = np.array(f1score_per_class)
    # f1score_per_class[np.isnan(f1score_per_class)] = 0
    #
    # return {"Micro F1-score": micro_f1score,
    #         "Macro F1-score": np.mean(macro_f1score),
    #         "Weighted F1-score": np.sum(weighted_f1score),
    #         "F1-score per class": np.array(f1score_per_class)}

    # def iu(self):
    #     hist = self.confusion_matrix
    #     intersection_over_union = np.diag(hist) / (
    #         hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    #     )
    #     mean_intersection_over_union = np.nanmean(intersection_over_union)
    #     intersection_over_union_per_class = dict(
    #         zip(range(self.num_classes), intersection_over_union)
    #     )
    #
    #     return {
    #         "intersection_over_union": intersection_over_union,
    #         "intersection_over_union_per_class": intersection_over_union_per_class,
    #     }
