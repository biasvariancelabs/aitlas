"""Base classes for change detection"""
import collections
import logging

import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm


from ..utils import DiceLoss
from .metrics import SegmentationRunningScore
from .models import BaseModel
from .schemas import BaseSegmentationClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, 
                 weight_focal: float = 0.5, 
                 weight_dice: float = 0.5,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 mode: str = "binary"):
        """
        Hybrid loss: CrossEntropy + Dice
        Args:
            weight_focal: weight for Focal component
            weight_dice: weight for Dice component
            mode: 'binary', 'multiclass', or 'multilabel'
        """
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(mode=mode, alpha=alpha, gamma=gamma)
        self.dice_loss = smp.losses.DiceLoss(mode=mode, from_logits=True)
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, y_pred, y_true):
        y_true_argmax = torch.argmax(y_true, dim=1)
        focal = self.focal_loss(y_pred, y_true_argmax)
        dice = self.dice_loss(y_pred, y_true_argmax)
        return self.weight_focal * focal + self.weight_dice * dice


class BaseChangeDetection(BaseModel):
    """Base class for a change detection model.
    """

    schema = BaseSegmentationClassifierSchema # Can reuse the segmentation schema for now

    def __init__(self, config):
        super().__init__(config)
        self.running_metrics = SegmentationRunningScore(self.num_classes, self.device)

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        self.model.train()
        running_loss = 0.0
        running_items = 0
        total_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, desc="training")):
            # get the inputs; data is a list of [(image1, image2), mask]
            inputs, labels = data
            image1, image2 = inputs
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self(image1, image2)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss.item() * image1.size(0)
            running_items += image1.size(0)
            total_loss += loss.item() * image1.size(0)
            if (i % iterations_log == iterations_log - 1):
                logging.info(
                    f"[{epoch + 1}, {i + 1}], loss: {running_loss / running_items : .5f}"
                )
                running_loss = 0.0
                running_items = 0

        total_loss = total_loss / len(dataloader.dataset)
        logging.info(
            f"epoch: {epoch + 1}, loss: {total_loss: .5f}"
        )
        return total_loss

    def predict_output_per_batch(self, dataloader, description):
        """Run predictions on a dataloader and return inputs, outputs, labels per batch"""
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc=description)):
                inputs, labels = data
                image1, image2 = inputs
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                labels = labels.to(self.device)

                outputs = self(image1, image2)

                if isinstance(outputs, collections.abc.Mapping):
                    outputs = outputs["out"]

                yield inputs, outputs, labels

    def get_predicted(self, outputs, threshold=None):
        """Get predicted classes from the model outputs."""
        predicted_probs = torch.sigmoid(outputs)
        predicted = (
            predicted_probs >= (threshold if threshold else self.config.threshold)
        ).long()
        return predicted_probs, predicted

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate)

    def load_criterion(self):
        """Load the loss function"""
        return CombinedFocalDiceLoss()

    def load_lr_scheduler(self, optimizer):
        """Load the learning rate scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.1, min_lr=1e-6
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)