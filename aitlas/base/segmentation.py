import logging

import torch
import torch.optim as optim

from torch import nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
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
                 mode: str = "multiclass"):
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

class BaseSegmentationClassifier(BaseModel):
    """Base class for a segmentation classifier.
    """

    schema = BaseSegmentationClassifierSchema

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = SegmentationRunningScore(self.num_classes, self.device)

    def get_predicted(self, outputs, threshold=None):
        """Get predicted classes from the model outputs.

        :param outputs: Model outputs with shape (batch_size, num_classes, H, W).
        :type outputs: torch.Tensor
        :param threshold: The threshold for classification, defaults to None. Not used in multiclass.
        :type threshold: float, optional
        :return: tuple containing the probabilities and predicted classes
        :rtype: tuple
        """
        num_classes = outputs.shape[1]
        predicted_probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(predicted_probs, dim=1)
        predicted_onehot = F.one_hot(predicted, num_classes=num_classes).permute(0, 3, 1, 2)
        return predicted_probs, predicted_onehot

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate)

    def load_criterion(self):
        """Load the loss function"""
        return CombinedFocalDiceLoss()

    def load_lr_scheduler(self, optimizer):
        """Load the learning rate scheduler"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, factor=0.1, min_lr=1e-6
        )
