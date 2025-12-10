"""Base classes for change detection"""
import collections
import logging
import copy

import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


import segmentation_models_pytorch as smp
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
        if self.focal_loss.mode == "multiclass":
            # if y_true is one-hot, convert to class indices
            if y_true.dim() == 4 and y_true.shape[1] > 1:
                y_true = torch.argmax(y_true, dim=1)
        
        focal = self.focal_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
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

    def evaluate_model(
        self,
        dataloader,
        criterion=None,
        description="testing on validation set",
    ):
        """
        Evaluates the current model against the specified dataloader for the specified metrics
        :param dataloader: The dataloader to evaluate against
        :param metrics: list of metric keys to calculate
        :criterion: Criterion to calculate loss
        :description: What to show in the progress bar
        :return: tuple of (metrics, y_true, y_pred)
        """
        self.model.eval()

        # initialize loss if applicable
        total_loss = 0.0

        for inputs, outputs, labels in self.predict_output_per_batch(
            dataloader, description
        ):
            image1, image2 = inputs
            if criterion:
                batch_loss = criterion(outputs, labels)
                total_loss += batch_loss.item() * image1.size(0)

            predicted_probs, predicted = self.get_predicted(outputs)

            if (
                len(labels.shape) == 1
            ):  # if it is multiclass, then we need one hot encoding for the predictions
                one_hot = torch.zeros(labels.size(0), self.num_classes)
                predicted = predicted.reshape(predicted.size(0))
                one_hot[torch.arange(labels.size(0)), predicted.type(torch.long)] = 1
                predicted = one_hot
                predicted = predicted.to(self.device)

            self.running_metrics.update(
                labels.type(torch.int64), predicted.type(torch.int64), predicted_probs
            )

        if criterion:
            total_loss = total_loss / len(dataloader.dataset)

        return total_loss

    def predict_masks(
        self,
        image1=None,  # Pre-event image
        image2=None,  # Post-event image
        labels=None,
        data_transforms=None,
        description="running prediction for single image pair",
    ):
        """
        Predicts using a model for a specified image pair and plots the predicted masks.

        :param image1: The pre-event image (numpy array HxWxC)
        :param image2: The post-event image (numpy array HxWxC)
        :param labels: List of labels for the masks
        :param data_transforms: Transformations to apply to the images
        :return: Plot of the predicted masks
        :rtype: matplotlib.figure.Figure
        """
        if image1 is None or image2 is None:
            raise ValueError("Both image1 and image2 must be provided for prediction.")

        original_image1 = copy.deepcopy(image1)
        original_image2 = copy.deepcopy(image2)

        self.model.eval()

        # Apply data transformations. Synchronized for both images.
        if data_transforms:
            # Stack images to ensure same random transforms are applied
            combined_images = np.concatenate([image1, image2], axis=-1)  # HxWxC -> HxWx(2C)
            transformed_combined = data_transforms(combined_images)

            # Split back into two images (C'xH'xW' -> (C'/2)xH'xW', (C'/2)xH'xW')
            # Assuming original images are 3 channels and transforms make it C'xH'xW'
            channels = image1.shape[2]  # original channel count
            inputs1, inputs2 = torch.split(transformed_combined, channels, dim=0)

        else:
            # If no transforms, convert to tensor and add batch dim
            # Assuming HxWxC numpy array, convert to 1xCxHxW tensor
            inputs1 = torch.from_numpy(image1.transpose(2, 0, 1)).unsqueeze(0)
            inputs2 = torch.from_numpy(image2.transpose(2, 0, 1)).unsqueeze(0)

        inputs1 = inputs1.to(self.device)
        inputs2 = inputs2.to(self.device)

        outputs = self(inputs1, inputs2)  # Model inference with two images

        # Check if outputs is OrderedDict for segmentation (from BaseModel)
        if isinstance(outputs, collections.abc.Mapping):
            outputs = outputs["out"]

        predicted_probs, predicted = self.get_predicted(outputs)
        predicted = list(predicted.cpu().detach().numpy())

        """Display image pair and predicted masks from model"""
        fig = plt.figure(figsize=(15, 7))  # Adjust figsize for three plots (image1, image2, mask)

        # Plot pre-event image
        plt.subplot(1, len(labels) + 2, 1)  # +2 for image1, image2, then labels
        plt.imshow(original_image1)  # Use original for display
        plt.title("Pre-event Image")
        plt.axis("off")

        # Plot post-event image
        plt.subplot(1, len(labels) + 2, 2)
        plt.imshow(original_image2)  # Use original for display
        plt.title("Post-event Image")
        plt.axis("off")

        # Plot masks
        for i in range(len(labels)):
            plt.subplot(1, len(labels) + 2, i + 3)  # Offset by 3 for image1, image2, and 1-based indexing
            plt.imshow(
                predicted[0][i].astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255
            )
            plt.title(labels[i])
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        return fig

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