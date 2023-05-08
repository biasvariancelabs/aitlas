import logging

import torch
import torch.optim as optim
import torchvision
from tqdm import tqdm

from ..utils import current_ts
from .metrics import ObjectDetectionRunningScore
from .models import BaseModel
from .schemas import BaseObjectDetectionSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseObjectDetection(BaseModel):

    """
    BaseObjectDetection

    This class extends the functionality of the BaseModel class by adding object detection specific functionality.
    It implements several functions required for training an object detection model, such as:

        Running object detection metrics during training.
        Non-Maximum Suppression (NMS) to get the final predictions.
        Loading an optimizer, loss function, and learning rate scheduler.
        Training a single epoch.

    Attributes:

        schema (BaseObjectDetectionSchema): Schema class to validate the configuration.
        log_loss (bool): Flag to log the loss during training.
        running_metrics (ObjectDetectionRunningScore): Running object detection metrics.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Gamma for the learning rate scheduler.

    Methods:

        get_predicted(outputs, threshold=0.3): Apply NMS to get the final predictions from the model outputs.
        load_optimizer(): Load an optimizer.
        load_criterion(): Load a loss function.
        load_lr_scheduler(optimizer): Load a learning rate scheduler.
        train_epoch(epoch, dataloader, optimizer, criterion, iterations_log): Train the model for a single epoch.

    """

    schema = BaseObjectDetectionSchema
    log_loss = True

    def __init__(self, config):
        super().__init__(config)

        self.running_metrics = ObjectDetectionRunningScore(
            self.num_classes, self.device
        )
        self.step_size = self.config.step_size
        self.gamma = self.config.gamma

    def get_predicted(self, outputs, threshold=0.3):

        # apply nms and return the indices of the bboxes to keep
        final_predictions = []
        for output in outputs:
            keep = torchvision.ops.nms(output["boxes"], output["scores"], threshold)

            final_prediction = output
            final_prediction["boxes"] = final_prediction["boxes"][keep]
            final_prediction["scores"] = final_prediction["scores"][keep]
            final_prediction["labels"] = final_prediction["labels"][keep]
            final_predictions.append(final_prediction)

        return final_predictions

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate)

    def load_criterion(self):
        """Load the loss function"""
        return None

    def load_lr_scheduler(self, optimizer):
        #return torch.optim.lr_scheduler.StepLR(
        #    optimizer, step_size=self.step_size, gamma=self.gamma
        #)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, min_lr=1e-6)

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        """Train the model for a single epoch.

        :param epoch: The current epoch number.
        :param dataloader: The data loader for the training set.
        :param optimizer: The optimizer.
        :param criterion: The loss function.
        :param iterations_log: The number of iterations after which to log the loss.

        Returns:

        float: The average loss over the entire epoch.

        """
        start = current_ts()
        running_loss = 0.0
        total_loss = 0.0

        self.model.train()
        for i, data in enumerate(tqdm(dataloader, desc="training")):
            inputs, targets = data

            inputs = list(
                image.type(torch.FloatTensor).to(self.device) for image in inputs
            )
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # zero the parameter gradients
            if isinstance(optimizer, tuple):
                for opt in optimizer:
                    opt.zero_grad()
            else:
                optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self(inputs, targets)
            loss = sum(loss for loss in outputs.values())
            loss.backward()

            # perform a single optimization step
            if isinstance(optimizer, tuple):
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()

            # log statistics
            running_loss += loss.item() * len(inputs)
            total_loss += loss.item() * len(inputs)

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

    def predict_output_per_batch(self, dataloader, description):
        """Run predictions on a dataloader and return inputs, outputs, targets per batch"""
        """_summary_

        :yield: _description_
        :rtype: _type_
        """
        # turn on eval mode
        self.model.eval()

        # run predictions
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc=description)):
                inputs, targets = data
                inputs = list(
                    image.type(torch.FloatTensor).to(self.device) for image in inputs
                )
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                outputs = self(inputs, targets)

                yield inputs, outputs, targets

    def evaluate_model(
        self, dataloader, criterion=None, description="testing on validation set",
    ):

        self.model.eval()

        for inputs, outputs, targets in self.predict_output_per_batch(
            dataloader, description
        ):

            predicted = self.get_predicted(outputs)
            self.running_metrics.update(predicted, targets)

        return 1 - self.running_metrics.get_scores(self.metrics)[0]['map']
