import dill
import numpy as np
import torch
from ignite.metrics import confusion_matrix
from ignite.metrics.multilabel_confusion_matrix import MultiLabelConfusionMatrix
from sklearn.metrics import average_precision_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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

    def __getstate__(self):
        state = self.__dict__.copy()
        state["confusion_matrix"] = dill.dumps(state["confusion_matrix"])
        return state

    def __setstate__(self, state):
        new_state = state
        new_state["confusion_matrix"] = dill.loads(state["confusion_matrix"])
        self.__dict__.update(new_state)

    def update(self, y_true, y_pred, y_prob=None):
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

    def weights(self):
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
        per_class = (
            2
            * precision["Precision per Class"]
            * recall["Recall per Class"]
            / (precision["Precision per Class"] + recall["Recall per Class"] + 1e-15)
        )

        return {
            "F1_score Micro": float(micro),
            "F1_score Macro": np.mean(per_class),
            "F1_score Weighted": np.sum(self.weights() * per_class),
            "F1_score per Class": per_class,
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
        return {"Accuracy": float(accuracy)}

    def weights(self):
        cm = self.get_computed()
        return (cm.sum(dim=1) / cm.sum()).numpy()

    def recall(self):
        cm = self.get_computed()
        micro = cm.diag().sum() / (cm.sum() + 1e-15)  # same as accuracy for multiclass
        macro = (
            cm.diag() / (cm.sum(dim=1) + 1e-15)
        ).mean()  # same as average accuracy in breizhcrops
        weighted = (
            (cm.diag() / (cm.sum(dim=1) + 1e-15))
            * ((cm.sum(dim=1)) / (cm.sum() + 1e-15))
        ).sum()
        per_class = cm.diag() / (cm.sum(dim=1) + 1e-15)

        return {
            "Recall Micro": float(micro),
            "Recall Macro": float(macro),
            "Recall Weighted": float(weighted),
            "Recall per Class": per_class.numpy(),
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
            "Precision Micro": float(micro),
            "Precision Macro": float(macro),
            "Precision Weighted": float(weighted),
            "Precision per Class": per_class.numpy(),
        }

    def iou(self):
        cm = self.get_computed()
        iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)

        return {"IOU": iou.tolist(), "mIOU": float(iou.mean())}

    def kappa(self):
        cm = self.get_computed()
        N = cm.shape[0]

        act_hist = cm.sum(axis=1)

        pred_hist = cm.sum(axis=0)

        num_samples = cm.sum()

        total_agreements = cm.diag().sum()
        agreements_chance = (act_hist * pred_hist) / num_samples
        agreements_chance = agreements_chance.sum()
        kappa = (total_agreements - agreements_chance) / (
            num_samples - agreements_chance
        )
        return {"Kappa metric": kappa}


class MultiLabelRunningScore(RunningScore):
    """Calculates a confusion matrix for multi-labelled, multi-class data in addition to the """

    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        self.confusion_matrix = MultiLabelConfusionMatrix(
            num_classes=self.num_classes, device=self.device,
        )
        self.list_y_prob = []
        self.list_y_true = []

    def reset(self):
        """Reset the confusion matrix and list of probabilities"""
        self.confusion_matrix.reset()
        self.list_y_prob = []
        self.list_y_true = []

    def update(self, y_true, y_pred, y_prob=None):
        """Updates stats on each batch"""
        self.confusion_matrix.update((y_pred, y_true))
        self.list_y_prob.extend(y_prob.tolist())
        self.list_y_true.extend(y_true.tolist())

    def map(self):
        return {"mAP": average_precision_score(np.array(self.list_y_true), np.array(self.list_y_prob))}

    def accuracy(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)

        accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + 1e-15)
        accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + 1e-15)

        return {"Accuracy": accuracy, "Accuracy per Class": accuracy_per_class}

    def precision(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)
        micro = tp_total / (tp_total + fp_total + 1e-15)
        per_class = tp / (tp + fp + 1e-15)
        macro = np.mean(per_class)
        weighted = np.sum(per_class * self.weights())
        return {
            "Precision Micro": float(micro),
            "Precision Macro": macro,
            "Precision Weighted": weighted,
            "Precision per Class": per_class,
        }

    def weights(self):
        tp, tn, fp, fn = self.get_outcomes()
        weights = (tp + fn) / self.get_samples()
        return weights

    def recall(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)
        micro = tp_total / (tp_total + fn_total + 1e-15)
        per_class = tp / (tp + fn + 1e-15)
        macro = np.mean(per_class)
        weighted = np.sum(per_class * self.weights())
        return {
            "Recall Micro": float(micro),
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

        if total:  # sum it all if we need to calculate the totals
            tp, tn, fp, fn = tp.sum(), tn.sum(), fp.sum(), fn.sum()

        return tp.numpy(), tn.numpy(), fp.numpy(), fn.numpy()

    def count(self):
        tp, tn, fp, fn = self.get_outcomes(True)
        return tp + tn + fp + fn

    def get_samples(self):
        cm = self.confusion_matrix.compute().cpu().detach().numpy()
        return np.sum(cm[:, 1, 0]) + np.sum(cm[:, 1, 1])

    def iou(self):
        tp, tn, fp, fn = self.get_outcomes()
        tp_total, tn_total, fp_total, fn_total = self.get_outcomes(total=True)

        iou_per_class = tp / (tp + fp + fn + 1e-15)
        iou = tp_total / (tp_total + fp_total + fn_total + 1e-15)

        return {
            "IOU": float(iou),
            "IOU mean": np.mean(iou_per_class),
            "IOU per Class": iou_per_class,
        }


class SegmentationRunningScore(MultiLabelRunningScore):
    """Calculates a metrics for semantic segmentation"""

    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)

    def update(self, y_true, y_pred, y_prob=None):
        """Updates stats on each batch"""
        self.confusion_matrix.update((y_pred, y_true))


class ObjectDetectionRunningScore(object):
    """Calculates a metrics for object detection"""

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    def update(self, preds, target):
        """Updates stats on each batch"""
        self.metric.update(preds, target)

    def reset(self):
        """Reset the confusion matrix"""
        self.metric.reset()

    def map(self):
        """Returns the specified metrics"""
        results = self.metric.compute()
        dict_results = {}
        for key, value in results.items():
            if len(list(value.size())):
                dict_results[key] = list(value)
            else:
                dict_results[key] = float(value)
        return dict_results

    def get_scores(self, metrics):
        """Returns the specified metrics"""
        result = []
        for metric in metrics:
            result.append(getattr(self, metric)())
        return result

