import dill
import numpy as np
import torch
import torchvision

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
            / (precision["Precision per Class"] + recall["Recall per Class"])
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
        ''' At the moment, the IoU threshold is fixed at 0.5'''
        self.iou_threshold = 0.5

        super().__init__(num_classes, device)
        
        # to store average precisions per class
        self.average_precisions = []

        # a value for numerical stability
        self.epsilon = 1e-6

        # keep a TP and FP tensor for each class separately
        self.TP_per_class = []
        self.FP_per_class = []
        # they all start with zero true boxes
        self.total_true_boxes_per_class = torch.zeros((num_classes))

    def update (self, y_true, y_pred):
        '''
            y_true = [[image_idx, class, 1.0, x1, y1, x2, y2]]
            y_pred = [[image_idx, pred_class, score, x1, y1, x2, y2]]
        '''

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        for c in range(self.num_classes):
            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            detections = y_pred[np.where(y_true[:, 1] == c)[0], :]
            ground_truths = y_true[np.where(y_true[:, 1] == c)[0], :]

            # find the amount of bboxes for each training example
            image_ind, counts = np.unique(ground_truths[:, 0], return_counts = True)

            # reformat these into a dictionary with the following shape
            # amount_bboxes = {image_idx: torch.tensor(num_true_bboxes)}
            amount_bboxes = {}
            for image_idx, count in zip(image_ind, counts):
                amount_bboxes[image_idx] = torch.zeros(count)

            # sort the detections in descending order based on the score
            detections = detections[np.argsort(detections[:, 2])[::-1]]
            self.TP_per_class.append(torch.zeros((len(detections))))
            self.FP_per_class.append(torch.zeros((len(detections))))
            self.total_true_boxes_per_class[c] = ground_truths.shape[0]

            # iterate through the images and calculate the number of TP and FP
            for image_idx in image_ind:
                # holds the indices of the detections in the TP_per_class array
                image_detections_ind = np.where(detections[:, 0]==image_idx)[0]

                image_detections = detections[image_detections_ind, :]
                image_gts = ground_truths[np.where(ground_truths[:, 0]==image_idx)[0], :]

                num_detections = image_detections.shape[0]

                # ious have the following shape: [NxM], where N is the number of detections
                # and M is the number of groundtruths
                ious = tochvision.ops.box_iou(image_detections[3:], image_gts[3:]).numpy()
                
                detection_best_iou = np.argmax(ious, axis=1)
                detection_potential_gt_idx = np.amax(ious, axis=1)

                # if the iou is lower than the threshold than than the detection is a false positive
                detection_potential_gt_idx[np.where(detection_best_iou<self.iou_threshold)] = -1

                for gt_idx in range(image_gts.shape[0]):
                    gt_ind = np.where(detection_potential_gt_idx == gt_idx)[0]
                    idx = gt_ind[0]
                    
                    if gt_ind.shape[0]>1:
                        rest = gt_ind[1:]
                        self.FP_per_class[c][image_detections_ind[rest]] = 1
                    
                    self.TP_per_class[c][image_detections_ind[idx]] = 1

    def reset (self):
        '''reset the state of all internal variables'''
        self.iou_threshold = 0.5
        
        # to store average precisions per class
        self.average_precisions = []

        # a value for numerical stability
        self.epsilon = 1e-6

        # keep a TP and FP tensor for each class separately
        self.TP_per_class = []
        self.FP_per_class = []
        # they all start with zero true boxes
        self.total_true_boxes_per_class = torch.zeros((self.num_classes))

    def f1_score(self):
        pass

    def precision(self):
        pass
    
    def recall(self):
        pass

    def mAP(self):
        for class_idx in range(self.num_classes):
            TP = self.TP_per_class[class_idx]
            FP = self.FP_per_class[class_idx]
            total_true_bboxes = self.total_true_boxes_per_class[class_idx]

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + self.epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + self.epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            self.average_precisions.append(torch.trapz(precisions, recalls))
        
        return sum(self.average_precisions) / len(self.average_precisions)