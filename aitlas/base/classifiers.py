import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import current_ts, stringify
from .datasets import BaseDataset
from .models import BaseModel
from .schemas import BaseClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseMulticlassClassifier(BaseModel):
    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)

    def fit(
        self,
        dataset: BaseDataset = None,
        epochs: int = 100,
        model_directory: str = None,
        save_epochs: int = 10,
        iterations_log: int = 100,
        resume_model: str = None,
        run_id: str = None,
        **kwargs,
    ):
        from ..metrics import F1Score

        logging.info("Starting training.")

        start_epoch = 0
        start = current_ts()

        # load loss, optimizer and lr scheduler
        self.criterion = self.load_criterion()
        self.optimizer = self.load_optimizer()
        self.lr_scheduler = self.load_lr_scheduler()

        # load the model if needs to resume training
        if resume_model:
            start_epoch, loss, start, run_id = self.load_model(
                resume_model, self.optimizer
            )

        # allocate device
        self.allocate_device()

        # start logger
        self.writer = SummaryWriter(os.path.join(model_directory, run_id))

        # get data loaders
        train_loader = dataset.train_loader()
        val_loader = dataset.val_loader()

        for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
            loss = self.train_epoch(
                epoch, train_loader, self.optimizer, self.criterion, iterations_log
            )
            self.writer.add_scalar("Loss/train", loss, epoch + 1)
            if epoch % save_epochs == 0:
                self.save_model(
                    model_directory, epoch, self.optimizer, loss, start, run_id
                )

            # adjust learning rate if needed
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # evaluate against a validation if there is one
            if val_loader:
                val_eval, y_true, y_pred, val_loss = self.evaluate_model(
                    val_loader, metrics=[F1Score], criterion=self.criterion
                )
                logging.info(stringify(val_eval))
                self.writer.add_scalar("Loss/val", val_loss, epoch + 1)

                self.log_additional_metrics(
                    val_eval,
                    y_true,
                    y_pred,
                    val_loss,
                    dataset,
                    model_directory,
                    run_id,
                    epoch,
                )

        self.writer.close()
        logging.info(f"finished training. training time: {current_ts() - start}")

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        start = current_ts()
        running_loss = 0.0
        total_loss = 0.0
        total = 0

        self.model.train()
        for i, data in enumerate(tqdm(dataloader, desc="training")):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.predict(inputs.to(self.device))
            loss = criterion(outputs, labels.to(self.device))
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss.item()
            total_loss += running_loss

            if (
                i % iterations_log == iterations_log - 1
            ):  # print every iterations_log mini-batches
                logging.info(
                    f"[{epoch + 1}, {i + 1}], loss: {running_loss / iterations_log : .5f}"
                )
                running_loss = 0.0

            total += 1

        total_loss = total_loss / (total * iterations_log)
        logging.info(
            f"epoch: {epoch + 1}, time: {current_ts() - start}, loss: {total_loss: .5f}"
        )
        return total_loss

    def predict(self, *input, **kwargs):
        return self.model(*input)

    def evaluate(
        self, dataset: BaseDataset = None, model_path: str = None, metrics: list = (),
    ):
        # load the model
        self.load_model(model_path)

        # get test data loader
        dataloader = dataset.test_loader()

        # evaluate model on data
        result = self.evaluate_model(dataloader, metrics)

        return result

    def evaluate_model(self, dataloader, metrics=(), criterion=None):
        """
        Evaluates the current model against the specified dataloader for the specified metrics
        :param dataloader:
        :param metrics: list of metric keys to calculate
        :return: tuple of (metrics, y_true, y_pred)
        """
        self.model.eval()

        # initialize counters
        y_true = []
        y_pred = []

        # initialize loss if applicable
        total_loss = 0.0
        total = 0

        # evaluate
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                outputs = self.predict(images.to(self.device))

                if criterion:
                    batch_loss = criterion(outputs, labels.to(self.device))
                    total_loss += batch_loss.item()
                    total += 1

                predicted = self.get_predicted(outputs)

                y_pred += list(predicted.cpu().numpy())
                y_true += list(labels.cpu().numpy())

        calculated_metrics = {}

        for metric_cls in metrics:
            metric = metric_cls()
            calculated_metrics[metric.name] = metric.calculate(y_true, y_pred)

        if criterion:
            total_loss = total_loss / total

        return (calculated_metrics, y_true, y_pred, total_loss)

    def get_predicted(self, outputs):
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def log_additional_metrics(
        self,
        val_eval,
        y_true,
        y_pred,
        val_loss,
        dataset,
        model_directory,
        run_id,
        epoch,
    ):
        from ..visualizations import confusion_matrix

        fig = confusion_matrix(
            dataset,
            y_true,
            y_pred,
            os.path.join(model_directory, run_id, f"cm_{epoch + 1}.png"),
        )
        self.writer.add_figure("Confusion matrix", fig, epoch + 1)

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.SGD(
            self.model.parameters(), lr=self.config.learning_rate, momentum=0.9
        )

    def load_criterion(self):
        """Load the loss function"""
        return nn.CrossEntropyLoss()

    def load_lr_scheduler(self):
        return None


class BaseMultilabelClassifier(BaseMulticlassClassifier):
    """The multilabel """

    def get_predicted(self, outputs):
        predicted_probs = torch.sigmoid(outputs)
        predicted = predicted_probs >= 0.5
        return predicted

    def log_additional_metrics(
        self,
        val_eval,
        y_true,
        y_pred,
        val_loss,
        dataset,
        model_directory,
        run_id,
        epoch,
    ):
        return True  # let's don't log anything for this
