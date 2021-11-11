import logging
import os
import sys

from tqdm import tqdm

import math
import numpy as np

import torch
import torch.optim as optim
from aitlas.base.models import BaseModel
from .schemas import BaseDetectionClassifierSchema
from aitlas.utils import current_ts

from .metrics import DetectionRunningScore

from .datasets import BaseDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s\n")

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
        '''
            Example output:

            {'boxes': tensor([[  5.7677, 108.3321,  15.2177, 113.5312],
                            [  6.7789, 106.9112,  16.5648, 116.3191],
                            [ 44.8771, 205.6994,  55.6011, 215.7120]], device='cuda:0'), 
            'labels': tensor([1, 2, 2], device='cuda:0'), 
            'scores': tensor([0.2266, 0.0687, 0.0514], device='cuda:0')}
        '''

        coco_formated_annotations = []

        for annot_idx, annot in enumerate(outputs):
            for box_id in range(annot['boxes'].shape[0]):
                coco_formated_annotations.append({
                    "image_id": annot_idx,
                    "category_id": int (annot['labels'][box_id].item()), 
                    "bbox": [float((annot['boxes'][box_id,2] - annot['boxes'][box_id,0])//2), 
                             float((annot['boxes'][box_id,3] - annot['boxes'][box_id,1])//2), 
                             float((annot['boxes'][box_id,2] - annot['boxes'][box_id,0])), 
                             float((annot['boxes'][box_id,3] - annot['boxes'][box_id,1]))],
                    "score": float(annot['scores'][box_id].item())
                })

        return None, coco_formated_annotations

    def get_groundtruth(self, labels):

        '''
            Example label:
            
            {'boxes': tensor([[ 79.2300,  36.1800, 123.6700,  80.6300],
                              [157.3000,  63.9400, 201.7400, 108.3900],
                              [ 93.1100, 112.5200, 137.5500, 156.9600]], device='cuda:0'), 
             'labels': tensor([2, 2, 2], device='cuda:0'), 
             'image_id': tensor([43], device='cuda:0'), 
             'area': tensor([1975.3577, 1975.3582, 1974.9142], device='cuda:0'), 
             'iscrowd': tensor([0, 0, 0], device='cuda:0')}
        '''
    
        coco_formated_annotations = {'annotations': [], "images": []}

        current_annot = 0
        for annot_idx, annot in enumerate(labels) :
            for box_id in range(annot['boxes'].shape[0]):
                coco_formated_annotations['annotations'].append({
                    "image_id": annot_idx, 
                    "category_id": int (annot['labels'][box_id].item()), 
                    "bbox": [float((annot['boxes'][box_id, 2] - annot['boxes'][box_id, 0])//2), 
                             float((annot['boxes'][box_id, 3] - annot['boxes'][box_id, 1])//2), 
                             float((annot['boxes'][box_id, 2] - annot['boxes'][box_id, 0])), 
                             float((annot['boxes'][box_id, 3] - annot['boxes'][box_id, 1]))],
                    "score": float(1.0), 
                    "id": current_annot, 
                    "iscrowd": annot['iscrowd'][box_id].item(), 
                    "area": annot['area'][box_id].item()
                })
                current_annot+=1

            coco_formated_annotations['images'].append(annot_idx)
        
        img_ids = np.unique(coco_formated_annotations['images'])
        coco_formated_annotations['images'] = [{"id": int(img_idx)} for img_idx in img_ids]

        coco_formated_annotations['categories'] = []

        # return the dictionary with the groundtruths as well as the new start_img_idx and the new annotation_start_idx
        return coco_formated_annotations

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

    def predict_with_output(self, dataset: BaseDataset = None, description="running prediction"):
        """
        Predicts using a model against for a specified dataset

        :return: 
        """

        # get the image_name order from the dataset
        img_names = dataset.get_img_names()
    
        # initialize an aggregation array 
        predictions = []
    
        # predict
        for inputs, outputs, labels in self.predict_output_per_batch(dataset.dataloader(), description):
            predictions.append(outputs.cpu())
        
        return img_names, predictions
