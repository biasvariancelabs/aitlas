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
        Configurable.__init__(self, config)
        super(BaseModel, self).__init__()

        self.model = nn.Module()

        device_name = "cpu"
        if self.config.use_cuda and torch.cuda.is_available():
            device_name = "cuda"

        self.device = torch.device(device_name)

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
