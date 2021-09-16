import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        """
        Dice Loss for image segmentation. Expects sigmoided inputs and binary targets.
        Implementation from: kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        """
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    ALPHA = 0.8
    GAMMA = 2

    def __init__(self):
        """
        Focal Loss for image segmentation. Expects sigmoided inputs and binary targets.
        Implementation from: kaggle.com/bigironsphere/loss-function-library-keras-pytorch
        """
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class ComboLoss(nn.Module):
    def __init__(self, loss_modules: list):
        """
        Helper module to combine (sum) multiple objectives with equal weights

        Args:
            loss_modules: List of objective functions
        """
        super().__init__()
        self.losses = loss_modules

    def forward(self, *args, **kwargs):
        loss_values = [l(*args, **kwargs) for l in self.losses]
        return sum(loss_values)


class DiceFocal(ComboLoss):
    def __init__(self):
        """
        Combination loss: DiceLoss() + FocalLoss()
        """
        super(DiceFocal, self).__init__([DiceLoss(), FocalLoss()])
