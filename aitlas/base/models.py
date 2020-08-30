import logging
import os
from shutil import copyfile

import torch
import torch.nn as nn

from ..utils import current_ts
from .config import Configurable
from .datasets import BaseDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BaseModel(nn.Module, Configurable):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        Configurable.__init__(self, config)

    def fit(
        self,
        dataset: BaseDataset = None,
        epochs: int = 100,
        model_directory: str = None,
        save_epochs: int = 10,
        iterations_log: int = 100,
        resume_model: str = None,
        **kwargs,
    ):
        """
        Trains the model on the given dataset. Saves the model on disk for reuse.
        """
        raise NotImplementedError

    def predict(self, *input, **kwargs):
        """
        Makes predictions for a given model and dataset.
        """
        raise NotImplementedError

    def evaluate(self, dataset: BaseDataset = None, model_path: str = None):
        """
        Evaluates a given model against a test dataset.
        """
        raise NotImplementedError

    def forward(self, *input, **kwargs):
        """
        Abstract method implementing the model. Extending classes should override this method.
        :return:  instance extending `nn.Module`
        """
        raise NotImplementedError

    def save_model(self, model_directory, epoch, optimizer, loss, start):
        """
        Saves the model on disk
        :param model_directory:
        :return:
        """
        if not os.path.isdir(model_directory):
            os.makedirs(model_directory)

        timestamp = current_ts()
        checkpoint = os.path.join(model_directory, f"checkpoint_{timestamp}.pth.tar")

        # create timestamped checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": self.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
                "start": start,
            },
            checkpoint,
        )

        # replace last checkpoint
        copyfile(checkpoint, os.path.join(model_directory, "checkpoint.pth.tar"))

    def load_model(self, file_path, optimizer=None):
        """Loads a model from a checkpoint"""
        if os.path.isfile(file_path):
            logging.info(f"=> loading checkpoint {file_path}")
            checkpoint = torch.load(file_path)

            self.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]
            loss = checkpoint["loss"]
            start = checkpoint["start"]

            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])

            logging.info(f"=> loaded checkpoint {file_path} at epoch {start_epoch}")
            return (start_epoch, loss, start)
        else:
            raise ValueError(f"No checkpoint found at {file_path}")
