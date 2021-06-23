import numpy as np
import torch

from collections import Counter
from aitlas.base import BaseMetric

class DetectionRunningScore(object):
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

    def update(self, pred_boxes, true_boxes, iou_threshold, box_format, num_classes):
        """Updates stats on each batch"""

        # update metrics
        for metric_cls in self.metrics:
            metric = metric_cls(device=self.device)
            calculated = metric.calculate(pred_boxes=pred_boxes, true_boxes=true_boxes, 
                                            iou_threshold = iou_threshold, 
                                            box_format = box_format,
                                            num_classes = num_classes)
            if isinstance(calculated, dict):
                if isinstance(self.calculated_metrics[metric.name], list):
                    self.calculated_metrics[metric.name] = {}
                for k, v in calculated.items():
                    if not k in self.calculated_metrics[metric.name]:
                        self.calculated_metrics[metric.name][k] = []
                    self.calculated_metrics[metric.name][k].append(v)
            else:
                self.calculated_metrics[metric.name].append(calculated)

        # # update confusion matrix
        # for lt, lp in zip(y_true, y_pred):
        #     self.confusion_matrix += self._fast_hist(
        #         lt.flatten(), lp.flatten(), self.num_classes
        #     )

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

class IoU(BaseMetric):
    name = "IoU"
    key = "iou"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred, eps=1e-7):
        total_score = 0.0
        for i, item in enumerate(y_true):
            predictions = torch.from_numpy(np.array(y_pred[i]))
            labels = torch.from_numpy(np.array(y_true[i]))

            predictions = predictions.to(self.device)
            labels = labels.to(self.device)

            intersection = torch.sum(labels * predictions)
            union = torch.sum(labels) + torch.sum(predictions) - intersection + eps
            total_score += (intersection + eps) / union

        return float(total_score / len(y_true))

class Accuracy(BaseMetric):
    name = "Accuracy"
    key = "accuracy"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def calculate(self, y_true, y_pred):
        total_score = 0.0
        for i, item in enumerate(y_true):
            predictions = torch.from_numpy(np.array(y_pred[i]))
            labels = torch.from_numpy(np.array(y_true[i]))

            predictions = predictions.to(self.device)
            labels = labels.to(self.device)

            tp = torch.sum(labels == predictions, dtype=predictions.dtype)
            total_score += tp / labels.view(-1).shape[0]

        return float(total_score / len(y_true))

class mAP (BaseMetric):
    name = "Mean Average Precision"
    key = "mAP"

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)
        self.method = None

    def intersection_over_union(self, boxes_preds, boxes_labels, box_format="midpoint"):
        '''
        Calculates intersection over union
        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
        Returns:
            tensor: Intersection over union for all examples
        '''
        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        elif box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # Need clamp(0) in case they do not intersect, then we want intersection to be 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)

    def calculate(self, pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes = 3):
        # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2]]
        # list storing all AP for respective classes
        average_precisions = []

        # used for numerical stability later on
        epsilon = 1e-6

        for c in range(num_classes):
            detections = []
            ground_truths = []

            # Go through all predictions and targets,
            # and only add the ones that belong to the
            # current class c
            for detection in pred_boxes:
                if detection[1] == c:
                    detections.append(detection)

            for true_box in true_boxes:
                if true_box[1] == c:
                    ground_truths.append(true_box)

            # find the amount of bboxes for each training example
            # Counter here finds how many ground truth bboxes we get
            # for each training example, so let's say img 0 has 3,
            # img 1 has 5 then we will obtain a dictionary with:
            # amount_bboxes = {0:3, 1:5}
            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            # We then go through each key, val in this dictionary
            # and convert to the following (w.r.t same example):
            # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            # sort by box probabilities which is index 2
            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)
            
            # If none exists for this class then we can safely skip
            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                # Only take out the ground_truths that have the same
                # training idx as detection
                ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
                ]

                num_gts = len(ground_truth_img)
                best_iou = 0

                for idx, gt in enumerate(ground_truth_img):
                    iou = self.intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        # true positive and add this bounding box to seen
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            # torch.trapz for numerical integration
            average_precisions.append(torch.trapz(precisions, recalls))

        return sum(average_precisions) / len(average_precisions)
