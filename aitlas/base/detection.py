import logging
import os
import sys

from tqdm import tqdm

import math

import torch
import torch.optim as optim
from aitlas.base.models import BaseModel
from .schemas import BaseDetectionClassifierSchema
from aitlas.utils import current_ts

from .metrics import DetectionRunningScore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

''' 
    Notes: 
    1. "train_epoch" is overwritten in order to handle the targets properly. 
'''

class BaseDetectionClassifier(BaseModel):
    # Define a schema which uses mAP as the base metric
    schema = BaseDetectionClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = DetectionRunningScore(self.num_classes, self.device)

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam([dict(params=self.model.parameters(), lr=0.0001),])

    def load_criterion(self):
        """Load the loss function"""
        return None

    def load_lr_scheduler(self):
        return None

    def get_predicted(self, outputs):
        count = 0
        pred_boxes = []
        for i in range (len(outputs)):
            for bbox_idx in range (outputs[i]['boxes'].shape[0]):
                pred_boxes.append([count, outputs[i]['labels'][bbox_idx], outputs[i]['scores'][bbox_idx], outputs[i]['boxes'][bbox_idx][0], 
                                                                                                                outputs[i]['boxes'][bbox_idx][1],
                                                                                                                outputs[i]['boxes'][bbox_idx][2],
                                                                                                                outputs[i]['boxes'][bbox_idx][3]])
            count += 1
        # the first return value is for compatibility with the BaseModel implementations
        return None, pred_boxes

    def get_groundtruth(self, labels):
        count = 0
        true_boxes = []
        for i in range (len(labels)):
            for bbox_idx in range (labels[i]['boxes'].shape[0]):
                true_boxes.append([count, labels[i]['labels'][bbox_idx], 1, labels[i]['boxes'][bbox_idx][0], 
                                                                                  labels[i]['boxes'][bbox_idx][1],
                                                                                  labels[i]['boxes'][bbox_idx][2],
                                                                                  labels[i]['boxes'][bbox_idx][3]])
            count += 1  
        return true_boxes

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        start = current_ts()
        running_loss = 0.0
        total_loss = 0.0

        self.model.train()
        for i, (images, targets) in enumerate(tqdm(dataloader, desc = "training")):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

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

