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
    '''
        Possible to-do:
            1. Run mAP at different thresholds
    '''

    def __init__(self, num_classes, device):
        ''' At the moment, the IoU threshold is fixed at 0.5'''
        self.iou_threshold = 0.5

        super().__init__(num_classes, device)
        
        # to store average precisions per class
        self.average_precisions = []

        # a value for numerical stability
        self.epsilon = 1e-6

        # keep a TP and FP tensor for each class separately
        # shape [num_detections_so_far X num_classes]
        # each update should append to the tensor of detections for the appropriate class
        self.TP_per_class = [torch.zeros((0)) for _ in range(self.num_classes)]
        self.FP_per_class = [torch.zeros((0)) for _ in range(self.num_classes)]
        # they all start with zero true boxes
        self.total_true_boxes_per_class = torch.zeros((num_classes))

    def update (self, y_true, y_pred):
        '''
            y_true = [[image_idx, class, 1.0, x1, y1, x2, y2]...[]]
            y_pred = [[image_idx, pred_class, score, x1, y1, x2, y2]...[]]
        '''

        # make sure these tensors are in RAM first
        y_true = [[x[0], x[1].cpu(), x[2], x[3].cpu(), x[4].cpu(), x[5].cpu(), x[6].cpu()] for x in y_true]
        y_pred = [[x[0], x[1].cpu(), x[2].cpu(), x[3].cpu(), x[4].cpu(), x[5].cpu(), x[6].cpu()] for x in y_pred]

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # iterate through all labels which could exist in this batch
        for c in range(self.num_classes):
            ''' 
                What happpens if the length of detections for this class is zero in this batch?
            '''
        
            # Go through all predictions and targets, and only select the ones that belong to the
            # current class c
            detections = y_pred[np.where(y_pred[:, 1] == c)[0], :]
            ground_truths = y_true[np.where(y_true[:, 1] == c)[0], :]

            # we should perform some checks before we continue... there is no point doing any calculations if there are 
            # no predictions or goundtruths for this class
            num_detections = detections.shape[0]
            num_gts = ground_truths.shape[0]

            # find the amount of gt_bboxes in each image
            gt_image_ind, gt_counts_per_image = np.unique(ground_truths[:, 0], return_counts = True)

            # reformat these into a dictionary with the following shape
            # amount_bboxes = {image_idx: torch.tensor(num_true_bboxes)}
            amount_bboxes = {}
            for image_idx, count in zip(gt_image_ind, gt_counts_per_image):
                amount_bboxes[image_idx] = torch.tensor(count)

            # sort the detections in DESCENDING order based on the score
            if detections.shape[0] != 0:
                detections = detections[np.argsort(detections[:, 2])[::-1]]
            
            # a local store for the TP and FP values which we will concatenate to the TP[FP]_per_class
            local_TP = torch.zeros((num_detections))
            local_FP = torch.zeros((num_detections))
            
            # increment the number of ground truths for this class
            self.total_true_boxes_per_class[c] += ground_truths.shape[0]

            # If no groundtruths for this class, all detections are false positives
            if not num_gts:
                if num_detections:
                    local_FP = torch.ones((num_detections))
                else:
                    continue
            if not num_detections:
                # this is ok because the mAP calculation happens at the end and not on each batch
                # this allows us to just add to the total_true_boxes_per_class and that way keep track of the FN across batches
                continue
            
            det_img_ind, det_counts = np.unique(detections[:, 0], return_counts = True)

            img_ind = list(set(det_img_ind).union(set(gt_image_ind)))
            # iterate through the images and calculate the number of TP and FP
            for image_idx in img_ind:
                # holds the indices of the detections in the local_TP[FP] array
                image_detections_ind = np.where(detections[:, 0]==image_idx)[0]
                image_gts_ind = np.where(ground_truths[:, 0]==image_idx)[0]

                num_img_detections = image_detections_ind.shape[0]
                num_img_gts = image_gts_ind.shape[0]

                image_detections = detections[image_detections_ind, :]
                image_gts = ground_truths[image_gts_ind, :]

                if not num_img_gts:
                    if num_img_detections:
                        local_FP[image_detections_ind] = 1
                    
                    continue
                
                if not num_img_detections:
                    continue

                # ious have the following shape: [NxM], where N is the number of detections
                # and M is the number of groundtruths
                detection_tensor = torch.from_numpy(image_detections[:, 3:].astype(np.float64))
                gt_tensor = torch.from_numpy(image_gts[:, 3:].astype(np.float64))

                ious = torchvision.ops.box_iou(detection_tensor, gt_tensor).numpy()
                
                detection_best_iou = np.amax(ious, axis=1)
                detection_potential_gt_idx = np.argmax(ious, axis=1)

                # if the iou is lower than the threshold than than the detection is a false positive
                detection_potential_gt_idx[np.where(detection_best_iou<self.iou_threshold)] = -1

                # iterate through all possible gt_indices
                for gt_idx in range(image_gts.shape[0]):

                    # the indices of image_detections that matches with the current ground_truth
                    match_ind = np.where(detection_potential_gt_idx == gt_idx)
                    
                    # check if we found a detection for that groundtruth at all
                    if match_ind[0].shape[0]:
                        match_ind = match_ind[0]
                    else:
                        continue

                    # find the index of the first match in the image_detections array
                    first_idx = match_ind[0]
                    
                    # if we find more than one match for this ground_truth
                    if match_ind.shape[0]>1:
                        # save the indices of all the other detections which were not the first
                        rest_ind = match_ind[1:]
                        # set all detections which were not the first for a ground_truth as FP
                        local_FP[image_detections_ind[rest_ind]] = 1
                    
                    local_TP[image_detections_ind[first_idx]] = 1
            
                # set all detections which were below the threshold as false positives
                local_FP[image_detections_ind[np.where(detection_potential_gt_idx== -1)[0]]] = 1

            # append to TP_per_class and FP_per_class
            self.TP_per_class[c] = torch.cat((self.TP_per_class[c], local_TP))
            self.FP_per_class[c] = torch.cat((self.FP_per_class[c], local_FP))

    def reset (self):
        ''' At the moment, the IoU threshold is fixed at 0.5'''
        self.iou_threshold = 0.5
        
        # to store average precisions per class
        self.average_precisions = []

        # a value for numerical stability
        self.epsilon = 1e-6

        # keep a TP and FP tensor for each class separately
        # shape [num_detections_so_far X num_classes]
        # each update should append to the tensor of detections for the appropriate class
        self.TP_per_class = [torch.zeros((0)) for _ in range(self.num_classes)]
        self.FP_per_class = [torch.zeros((0)) for _ in range(self.num_classes)]
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
        
        return {
            "mAP@0.5": sum(self.average_precisions[1:]) / len(self.average_precisions[1:]), 
            "AP@0.5 per Class": self.average_precisions[1:],
        }