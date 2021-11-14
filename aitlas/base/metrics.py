import dill
import numpy as np
import torch
import torchvision

from ignite.metrics import confusion_matrix
from ignite.metrics.multilabel_confusion_matrix import MultiLabelConfusionMatrix

from ..utils import COCO
from ..utils import COCOeval

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

    def update(self, y_true, y_pred, **kwargs):
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
            / (precision["Precision Micro"] + recall["Recall Micro"])
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
            "Recall Micro": micro,
            "Recall Macro": macro,
            "Recall Weighted": weighted,
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
            "Precision Micro": micro,
            "Precision Macro": macro,
            "Precision Weighted": weighted,
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
        micro = tp_total / (tp_total + fn_total)
        per_class = tp / (tp + fn)
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

        iou_per_class = tp / (tp + fp + fn)
        iou = tp_total / (tp_total + fp_total + fn_total)

        return {
            "IOU": float(iou),
            "IOU mean": np.mean(iou_per_class),
            "IOU per Class": iou_per_class.tolist(),
        }

class SegmentationRunningScore(RunningScore):
    """Calculates a metrics for semantic segmentation"""

    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        self.iou_per_class = torch.zeros(num_classes, dtype=torch.float64).to(
            self.device
        )
        self.f1_score_per_class = torch.zeros(num_classes, dtype=torch.float64).to(
            self.device
        )
        self.intersection_per_class = torch.zeros(num_classes, dtype=torch.float64).to(
            self.device
        )
        self.total_per_class = torch.zeros(num_classes, dtype=torch.float64).to(
            self.device
        )
        self.pixel_accuracy_per_class = torch.zeros(
            num_classes, dtype=torch.float64
        ).to(self.device)
        self.samples = 0

    def update(self, y_true, y_pred):
        """Updates metrics on each batch"""
        num_images, num_labels, h, w = y_true.shape
        self.samples += num_images
        for i in range(num_images):
            for j in range(num_labels):
                y_pred_local = y_pred[i, j, :, :].unsqueeze(0)
                y_true_local = y_true[i, j, :, :].unsqueeze(0)
                intersection = (y_pred_local & y_true_local).float().sum()
                union = (y_pred_local | y_true_local).float().sum()
                correct = (y_pred_local == y_true_local).int().sum()

                total = y_true_local.numel()
                trues = y_pred_local.float().sum() + y_true_local.float().sum()

                self.iou_per_class[j] += 1 if union == 0 else (intersection / union)
                self.f1_score_per_class[j] += (
                    1 if trues == 0 else (2 * intersection / trues)
                )
                self.pixel_accuracy_per_class[j] += correct / total

    def reset(self):
        """Reset the metrics"""
        self.iou_per_class = torch.zeros(self.num_classes, dtype=torch.float64).to(
            self.device
        )
        self.f1_score_per_class = torch.zeros(self.num_classes, dtype=torch.float64).to(
            self.device
        )
        self.intersection_per_class = torch.zeros(
            self.num_classes, dtype=torch.float64
        ).to(self.device)
        self.total_per_class = torch.zeros(self.num_classes, dtype=torch.float64).to(
            self.device
        )
        self.pixel_accuracy_per_class = torch.zeros(
            self.num_classes, dtype=torch.float64
        ).to(self.device)
        self.samples = 0

    def accuracy(self):
        self.pixel_accuracy_per_class = self.pixel_accuracy_per_class / self.samples
        return {
            "Accuracy mean": float(self.pixel_accuracy_per_class.mean()),
            "Accuracy per Class": self.pixel_accuracy_per_class.tolist(),
        }

    def f1_score(self):
        self.f1_score_per_class = self.f1_score_per_class / self.samples
        return {
            "F1 mean": float(self.f1_score_per_class.mean()),
            "F1 per Class": self.f1_score_per_class.tolist(),
        }

    def iou(self):
        self.iou_per_class = self.iou_per_class / self.samples
        return {
            "IOU mean": float(self.iou_per_class.mean()),
            "IOU per Class": self.iou_per_class.tolist(),
        }
class DetectionRunningScore(RunningScore):

    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        
        self.cocoGt = None
        self.cocoDt = None

        self.predictions = []
        self.groundtruths = {'annotations': [], 'images': [], 'categories': None}

    def reset (self):
        self.cocoGt = None
        self.cocoDt = None

        self.predictions = []
        self.groundtruths = {'annotations': [], 'images': [], 'categories': None}

    def correct_indices(self):
        predictions = [x['image_id'] for x in self.predictions]

        past_image_id = 0
        current_image_idx = 0

        for i in range (len(predictions)):
            if predictions[i] != past_image_id:
                current_image_idx += 1
            
            past_image_id = predictions[i]
            predictions[i] = current_image_idx
        
        for (i, corr_img_id) in zip(range(len(self.predictions)), predictions):
            self.predictions[i]['image_id'] = corr_img_id

        groundtruths = [x['image_id'] for x in self.groundtruths['annotations']]

        past_image_id = 0
        current_image_idx = 0

        for i in range (len(groundtruths)):
            if groundtruths[i] != past_image_id:
                current_image_idx += 1
            
            past_image_id = groundtruths[i]
            groundtruths[i] = current_image_idx

        for (i, corr_img_id) in zip(range(len(self.groundtruths['annotations'])), groundtruths):
            self.groundtruths['annotations'][i]['image_id'] = corr_img_id
            self.groundtruths['annotations'][i]['id'] = i

        image_indices = list(range(current_image_idx + 1))
        self.groundtruths['images'] = [{"id": int(img_idx)} for img_idx in image_indices]

    def add_categories(self):
        self.groundtruths['categories'] = [{"id": 0, "name": "background", "supercategory": "background"}]

        labels = [int (x['category_id']) for x in self.groundtruths['annotations']]
        labels = np.unique(labels)

        for label in labels:
            self.groundtruths['categories'].append({"id": label, "name": str(label), "supercategory": str(label)})

    def update (self, y_true, y_pred):

        self.predictions += y_pred
        
        self.groundtruths['categories'] = y_true['categories']

        for key in y_true.keys():
            if key != 'categories':
                self.groundtruths[key] += y_true[key]

    def cocoAP(self):

        self.correct_indices()
        self.add_categories()

        self.cocoGt=COCO(self.groundtruths)
        self.cocoDt=self.cocoGt.loadRes(self.predictions)

        self.cocoEvaluation = COCOeval(self.cocoGt, self.cocoDt,'bbox')
        self.cocoEvaluation.evaluate()
        self.cocoEvaluation.accumulate()
        self.cocoEvaluation.summarize()
        
        return self.cocoEvaluation.stats