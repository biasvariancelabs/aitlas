import logging
import os
import sys

from tqdm import tqdm

import math

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from aitlas.base.datasets import BaseDataset
from aitlas.base.models import BaseModel
from .schemas import BaseDetectionClassifierSchema
from aitlas.utils import current_ts

from .metrics import RunningScore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

''' 
    Notes: 
    1. The "fit" method had to be overwritten because "evaluate_model" does not return the loss. 
       This is because we still don't have an implementation of a loss function used by an object detection model.
    2. "train_epoch" and "predict_output_per_batch" are also overwritten in order to handle the targets properly. 
    3. "evaluate_model" is overwritten in order to avoid calculating loss as well as to prepare the outputs in a suitable format for the mAP metric.

    We also implement the mAP (Mean Average Precison) metric. As opposed to other metrics, in order to calculate mAP we need the predicted and groundtruth 
    boxes, as well as the iou threshold rather than the true and groundtruth labels. This means we have to change the "update" method in the RunningScore class. 
    In order to do that, we implement a version of the RunningScore class with **kwargs, to allow for calls to update functions with different signatures.
'''

class BaseDetectionClassifier(BaseModel):
    # Define a schema which uses mAP as the base metric
    schema = BaseDetectionClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = RunningScore(
            self.metrics, self.config.num_classes, self.device
        )

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam([dict(params=self.model.parameters(), lr=0.0001),])

    def load_criterion(self):
        """Load the loss function"""
        return None

    def load_lr_scheduler(self):
        return None

    def get_predicted(self, image_count, outputs):
        count = image_count
        pred_boxes = []
        for i in range (len(outputs)):
            for bbox_idx in range (outputs[i]['boxes'].shape[0]):
                pred_boxes.append([image_count, outputs[i]['labels'][bbox_idx], outputs[i]['scores'][bbox_idx], outputs[i]['boxes'][bbox_idx][0], 
                                                                                                                outputs[i]['boxes'][bbox_idx][1],
                                                                                                                outputs[i]['boxes'][bbox_idx][2],
                                                                                                                outputs[i]['boxes'][bbox_idx][3]])
            count += 1
        return pred_boxes, count

    def get_groundtruth(self, image_count, outputs, labels):
        count = image_count
        true_boxes = []
        for i in range (len(outputs)):
            for bbox_idx in range (labels[i]['boxes'].shape[0]):
                true_boxes.append([image_count, labels[i]['labels'][bbox_idx], 1, labels[i]['boxes'][bbox_idx][0], 
                                                                                  labels[i]['boxes'][bbox_idx][1],
                                                                                  labels[i]['boxes'][bbox_idx][2],
                                                                                  labels[i]['boxes'][bbox_idx][3]])
            count += 1  
        return true_boxes, count

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        start = current_ts()
        running_loss = 0.0
        total_loss = 0.0

        self.model.train()
        for i, (images, targets) in enumerate(tqdm(dataloader, desc = "training")):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.forward_train(images, targets)

            combined_loss = sum(loss for loss in loss_dict.values())

            loss_value = combined_loss.item()

            running_loss += combined_loss.item() * len(images)
            total_loss += combined_loss * len(images)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            if (
                i % iterations_log == iterations_log - 1
            ):  # print every iterations_log mini-batches
                logging.info(
                    f"[{epoch + 1}, {i + 1}], loss: {running_loss / iterations_log : .5f}"
                )
                running_loss = 0.0

        total_loss = total_loss / len(dataloader.dataset)
        logging.info(
            f"epoch: {epoch + 1}, time: {current_ts() - start}, loss: {total_loss: .5f}"
        )

        return total_loss

    def evaluate_model(self, dataloader, criterion=None, description="testing on validation set"):
        """
        Evaluates the current model against the specified dataloader for the specified metrics
        :param dataloader:
        :param metrics: list of metric keys to calculate
        :criterion: Criterion to calculate loss
        :description: What to show in the progress bar
        :return: tuple of (metrics, y_true, y_pred)
        """
        self.model.eval()
        
        pred_boxes = []
        true_boxes = []

        image_count = 0
        for inputs, outputs, labels in self.predict_output_per_batch(dataloader, description):
            formated_preds, count = self.get_predicted(image_count, outputs)
            pred_boxes += formated_preds
            
            formated_true, count = self.get_groundtruth(image_count, outputs, labels)
            true_boxes += formated_true

            image_count+=count


        self.running_metrics.update(true_boxes, pred_boxes,
                                    iou_threshold = 0.5, 
                                    box_format = 'corners', 
                                    num_classes = 3, 
                                    detection_clf = True,
                                    confusion_matrix = False)

        return 0

    def predict_output_per_batch(self, dataloader, description):
        """Run predictions on a dataloader and return inputs, outputs, labels per batch"""
        # turn on eval mode
        self.model.eval()

        # run predictions
        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(dataloader, desc = description)):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs= self.forward_eval(images)

                yield images, outputs, targets
