import collections
import logging
import os
from shutil import copyfile

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import current_ts, stringify
from .config import Configurable
from .datasets import BaseDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseModel(nn.Module, Configurable):
    def __init__(self, config):
        Configurable.__init__(self, config)
        super(BaseModel, self).__init__()

        self.model = nn.Module()

        device_name = "cpu"
        if self.config.use_cuda and torch.cuda.is_available():
            device_name = "cuda"

        self.device = torch.device(device_name)

    def fit(
        self,
        dataset: BaseDataset,
        epochs: int = 100,
        model_directory: str = None,
        save_epochs: int = 10,
        iterations_log: int = 100,
        resume_model: str = None,
        val_dataset: BaseDataset = None,
        run_id: str = None,
        metrics: tuple = (),
        **kwargs,
    ):
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
        train_loader = dataset.dataloader()
        val_loader = None
        if val_dataset:
            val_loader = val_dataset.dataloader()

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

            # evaluate against the train set
            calculated = self.evaluate_model(
                train_loader, metrics=metrics, criterion=self.criterion
            )
            self.log_metrics(calculated, "train", self.writer, epoch + 1)

            # evaluate against a validation set if there is one
            if val_loader:
                calculated = self.evaluate_model(
                    val_loader, metrics=metrics, criterion=self.criterion
                )
                self.log_metrics(calculated, "val", self.writer, epoch + 1)
                _, _, _, _, val_loss = calculated
                self.writer.add_scalar("Loss/val", val_loss, epoch + 1)

        self.writer.close()

        # save the model in the end
        self.save_model(model_directory, epochs, self.optimizer, loss, start, run_id)

        logging.info(f"finished training. training time: {current_ts() - start}")

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        start = current_ts()
        running_loss = 0.0
        total_loss = 0.0

        self.model.train()
        for i, data in enumerate(tqdm(dataloader, desc="training")):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.predict(inputs)

            # check if outputs is OrderedDict for segmentation
            if isinstance(outputs, collections.Mapping):
                outputs = outputs["out"]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss.item() * inputs.size(0)
            total_loss += loss.item() * inputs.size(0)

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

    def predict(self, *input, **kwargs):
        return self(*input)

    def evaluate(
        self, dataset: BaseDataset = None, model_path: str = None, metrics: list = (),
    ):
        # load the model
        self.load_model(model_path)

        # get test data loader
        dataloader = dataset.dataloader()

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
        y_pred_probs = []

        # initialize loss if applicable
        total_loss = 0.0

        # evaluate
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc="validation")):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.predict(inputs)

                # check if outputs is OrderedDict for segmentation
                if isinstance(outputs, collections.Mapping):
                    outputs = outputs["out"]

                if criterion:
                    batch_loss = criterion(outputs, labels)
                    total_loss += batch_loss.item() * inputs.size(0)

                predicted_probs, predicted = self.get_predicted(outputs)
                # print('Labels ', labels.cpu().detach().numpy())
                # print('Predicted ', predicted.cpu().detach().numpy())

                # y_pred_probs += list(predicted_probs.cpu().detach().numpy())
                # y_pred += list(predicted.cpu().detach().numpy())
                # y_true += list(labels.cpu().detach().numpy())

                y_pred_probs += list(predicted_probs)
                y_pred += list(predicted)
                y_true += list(labels)

        calculated_metrics = {}

        for metric_cls in metrics:
            metric = metric_cls()
            print(metric.calculate(y_true, y_pred))
            calculated_metrics[metric.name] = metric.calculate(y_true, y_pred)
            # for i, item in enumerate(y_true):
            #    calculated_metrics[metric.name] += metric.calculate(y_true[i], y_pred[i])
            # calculated_metrics[metric.name] /= len(y_true)

        if criterion:
            total_loss = total_loss / len(dataloader.dataset)

        return (calculated_metrics, y_true, y_pred, y_pred_probs, total_loss)

    def forward(self, *input, **kwargs):
        """
        Abstract method implementing the model. Extending classes should override this method.
        :return:  instance extending `nn.Module`
        """
        raise NotImplementedError

    def get_predicted(self, outputs):
        """Gets the output from the model and return the predictions
        :return: tuple in the format (probabilities, predicted classes/labels)
        """
        raise NotImplementedError("Please implement `get_predicted` for your model. ")

    def metrics(self):
        """Metrics we want to log for the model"""
        return ()

    def report(self, y_true, y_pred, y_prob, labels, **kwargs):
        """The report we want to generate for the model"""
        return ()

    def log_metrics(self, output, tag="train", writer=None, epoch=0):
        """Log the calculated metrics"""
        calculated_metrics, y_true, y_pred, y_probs, val_loss = output
        logging.info(stringify(calculated_metrics))
        if writer:
            for metric_name in calculated_metrics:
                metric = calculated_metrics[metric_name]
                if isinstance(metric, dict):
                    for sub in metric:
                        writer.add_scalar(
                            f"{metric_name}_{sub}/{tag}", metric[sub], epoch
                        )
                else:
                    writer.add_scalar(f"{metric_name}/{tag}", metric, epoch)

    def allocate_device(self, opts=None):
        """
        Put the model on CPU or GPU
        :return:
        """
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        return self.model

    def save_model(self, model_directory, epoch, optimizer, loss, start, run_id):
        """
        Saves the model on disk
        :param model_directory:
        :return:
        """
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)

        if not os.path.isdir(os.path.join(model_directory, run_id)):
            os.makedirs(os.path.join(model_directory, run_id))

        timestamp = current_ts()
        checkpoint = os.path.join(
            model_directory, run_id, f"checkpoint_{timestamp}.pth.tar"
        )

        # create timestamped checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
                "start": start,
                "id": run_id,
            },
            checkpoint,
        )

        # replace last checkpoint
        copyfile(checkpoint, os.path.join(model_directory, "checkpoint.pth.tar"))

    def load_model(self, file_path, optimizer=None):
        """Loads a model from a checkpoint"""
        if os.path.isfile(file_path):
            logging.info(f"Loading checkpoint {file_path}")
            checkpoint = torch.load(file_path)

            self.model.load_state_dict(checkpoint["state_dict"])
            self.allocate_device()

            start_epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            start = checkpoint["start"]
            run_id = checkpoint["id"]

            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])

            logging.info(f"Loaded checkpoint {file_path} at epoch {start_epoch}")
            return (start_epoch, loss, start, run_id)
        else:
            raise ValueError(f"No checkpoint found at {file_path}")

    def train_model(
        self,
        train_dataset: BaseDataset,
        epochs: int = 100,
        model_directory: str = None,
        save_epochs: int = 10,
        iterations_log: int = 100,
        resume_model: str = None,
        val_dataset: BaseDataset = None,
        run_id: str = None,
        metrics: tuple = (),
        **kwargs,
    ):
        return self.fit(
            dataset=train_dataset,
            epochs=epochs,
            model_directory=model_directory,
            save_epochs=save_epochs,
            iterations_log=iterations_log,
            resume_model=resume_model,
            run_id=run_id,
            metrics=metrics,
            **kwargs,
        )

    def train_and_evaluate_model(
        self,
        train_dataset: BaseDataset,
        epochs: int = 100,
        model_directory: str = None,
        save_epochs: int = 10,
        iterations_log: int = 100,
        resume_model: str = None,
        val_dataset: BaseDataset = None,
        run_id: str = None,
        metrics: tuple = (),
        **kwargs,
    ):
        return self.fit(
            dataset=train_dataset,
            epochs=epochs,
            model_directory=model_directory,
            save_epochs=save_epochs,
            iterations_log=iterations_log,
            resume_model=resume_model,
            val_dataset=val_dataset,
            run_id=run_id,
            metrics=metrics,
            **kwargs,
        )
