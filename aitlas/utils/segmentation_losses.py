"""Loss functions for image segmentation"""

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        """
        Dice Loss for image segmentation. Expects sigmoided inputs and binary targets.
        ..note:: Implementation from: kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        """
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    ALPHA = 0.8
    GAMMA = 2

    def __init__(self):
        """
        Focal Loss for image segmentation. Expects sigmoided inputs and binary targets.
        ..note:: Implementation from: kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        """
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class BCL(nn.Module):
    """
    batch-balanced contrastive loss for change detection (for STANet)
    no-change, 1
    change, -1
    """

    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 255] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label == 1).float()) + 0.0001
        neg_num = torch.sum((label == -1).float()) + 0.0001

        loss_1 = torch.sum((1 + label) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = (
            torch.sum(
                (1 - label)
                / 2
                * mask
                * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
            )
            / neg_num
        )
        loss = loss_1 + loss_2
        return loss
