"""Models base class.
This is the base class for all models. All models should subclass it. 
"""
import collections
import copy
import logging
import os
from shutil import copyfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import current_ts, save_best_model, stringify
from .config import Configurable
from .datasets import BaseDataset
from .schemas import BaseModelSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=10, min_delta=0):
        """BaseModel constructor

        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            logging.info(
                f"INFO: Early stopping counter {self.counter} of {self.patience}"
            )
            if self.counter >= self.patience:
                logging.info("INFO: Early stopping")
                self.early_stop = True


class BaseModel(nn.Module, Configurable):
    """Basic class abstracting a model. Contains methods for training,
    evaluation and also utility methods for loading, saving a model to storage.
    """

    schema = BaseModelSchema
    name = None
    log_loss = True

    def __init__(self, config=None):
        """BaseModel constructor

        :param config: Configuration object which specifies the details of the model, defaults to None.
        :type config: Config, optional
        """
        Configurable.__init__(self, config)
        super(BaseModel, self).__init__()

        self.model = nn.Module()

        device_name = "cpu"
        if self.config.use_cuda and torch.cuda.is_available():
            device_name = f"cuda:{self.config.rank}"

        self.device = torch.device(device_name)

        self.metrics = self.config.metrics
        self.num_classes = self.config.num_classes
        self.weights = (
            torch.tensor(self.config.weights, dtype=torch.float32)
            if self.config.weights
            else None
        )

    def prepare(self):
        """Prepare the model before using it. Loans loss criteria, optimizer, lr scheduler and early stopping."""

        # load loss, optimizer and lr scheduler
        self.criterion = self.load_criterion()
        self.optimizer = self.load_optimizer()
        self.lr_scheduler = self.load_lr_scheduler(self.optimizer)
        self.early_stopping = EarlyStopping()

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
        **kwargs,
    ):
        """Main method to train the model. It trains the model for the specified number of epochs and saves the model after every save_epochs. It also logs the loss after every iterations_log.

        :param dataset: Dataset object which contains the training data.
        :type dataset: aitlas.base.BaseDataset
        :param epochs: Number of epochs to train the model, defaults to 100
        :type epochs: int, optional
        :param model_directory: Location where the model checkpoints will be stored or should be loaded from, defaults to None
        :type model_directory: str, optional
        :param save_epochs: Number of epoch after a checkpoint is saved, defaults to 10
        :type save_epochs: int, optional
        :param iterations_log: Number of iteration after which the training status will be logged, defaults to 100
        :type iterations_log: int, optional
        :param resume_model: Whether or not to resume training a saved model, defaults to None
        :type resume_model: str, optional
        :param val_dataset: Dataset object which contains the validation data., defaults to None
        :type val_dataset: aitlas.base.BaseDataset, optional
        :param run_id: Optional id to idenfity the experiment, defaults to None
        :type run_id: str, optional
        :return: Returns the loss at the end of training.
        :rtype: float
        """
        logging.info("Starting training.")

        start_epoch = 0
        train_losses = []
        val_losses = []
        train_time_epoch = []
        total_train_time = 0

        best_loss = None
        best_epoch = None
        best_model = None

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
            start = current_ts()
            loss = self.train_epoch(
                epoch, train_loader, self.optimizer, self.criterion, iterations_log
            )
            train_time = current_ts() - start
            total_train_time += train_time
            train_time_epoch.append(train_time)

            self.writer.add_scalar("Loss/train", loss, epoch + 1)
            if epoch % save_epochs == 0:
                self.save_model(
                    model_directory, epoch, self.optimizer, loss, start, run_id
                )

            # evaluate against the train set
            self.running_metrics.reset()
            train_loss = self.evaluate_model(
                train_loader,
                criterion=self.criterion,
                description="testing on train set",
            )
            self.log_metrics(
                self.running_metrics.get_scores(self.metrics),
                dataset.get_labels(),
                "train",
                self.writer,
                epoch + 1,
            )

            # for object detection log the loss calculated during training, otherwise the loss calculated in eval mode
            if train_loss:
                train_losses.append(train_loss)
            else:
                train_losses.append(loss)

            # evaluate against a validation set if there is one
            if val_loader:
                self.running_metrics.reset()
                val_loss = self.evaluate_model(
                    val_loader,
                    criterion=self.criterion,
                    description="testing on validation set",
                )

                self.log_metrics(
                    self.running_metrics.get_scores(self.metrics),
                    dataset.get_labels(),
                    "val",
                    self.writer,
                    epoch + 1,
                )

                if self.log_loss:
                    if not best_loss or val_loss < best_loss:
                        best_loss = val_loss
                        best_epoch = epoch
                        best_model = copy.deepcopy(self.model)

                    # adjust learning rate if needed
                    if self.lr_scheduler:
                        if isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step(val_loss)
                        else:
                            self.lr_scheduler.step()

                    val_losses.append(val_loss)
                    self.early_stopping(val_loss)
                    if self.early_stopping.early_stop:
                        break

                    self.writer.add_scalar("Loss/val", val_loss, epoch + 1)
            else:
                if self.lr_scheduler and not isinstance(
                    self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.lr_scheduler.step()

        self.writer.close()

        # save the model in the end
        self.save_model(model_directory, epochs, self.optimizer, loss, start, run_id)

        # save the model with lowest validation loss
        if best_model:
            save_best_model(
                best_model,
                model_directory,
                best_epoch + 1,
                self.optimizer,
                best_loss,
                start,
                run_id,
            )

        logging.info(f"Train loss: {train_losses}")
        logging.info(f"Validation loss: {val_losses}")
        logging.info(f"Train time per epochs: {train_time_epoch}")
        logging.info(f"Finished training. training time: {total_train_time}")

        return loss

    def train_epoch(self, epoch, dataloader, optimizer, criterion, iterations_log):
        start = current_ts()
        running_loss = 0.0
        running_items = 0
        total_loss = 0.0

        self.model.train()
        for i, data in enumerate(tqdm(dataloader, desc="training")):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            if isinstance(optimizer, tuple):
                for opt in optimizer:
                    opt.zero_grad()
            else:
                optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self(inputs)

            # check if outputs is OrderedDict for segmentation
            if isinstance(outputs, collections.abc.Mapping):
                outputs = outputs["out"]

            loss = criterion(outputs, labels)
            loss.backward()

            # perform a single optimization step
            if isinstance(optimizer, tuple):
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()

            # log statistics
            running_loss += loss.item() * inputs.size(0)
            running_items += inputs.size(0)
            total_loss += loss.item() * inputs.size(0)

            if (
                i % iterations_log == iterations_log - 1
            ):  # print every iterations_log mini-batches
                logging.info(
                    f"[{epoch + 1}, {i + 1}], loss: {running_loss / running_items : .5f}"
                )
                running_loss = 0.0
                running_items = 0

        total_loss = total_loss / len(dataloader.dataset)
        logging.info(
            f"epoch: {epoch + 1}, time: {current_ts() - start}, loss: {total_loss: .5f}"
        )
        return total_loss

    def evaluate(
        self,
        dataset: BaseDataset = None,
        model_path: str = None,
    ):
        """
        Evaluate a model stored in a specified path against a given dataset

        :param dataset: the dataset to evaluate against
        :param model_path: the path to the model on disk
        :return:
        """
        # load the model
        self.load_model(model_path)

        # get test data loader
        dataloader = dataset.dataloader()

        # evaluate model on data
        result = self.evaluate_model(dataloader, description="testing on test set")

        return result

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
            if criterion:
                batch_loss = criterion(outputs, labels)
                total_loss += batch_loss.item() * inputs.size(0)

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

    def predict(
        self,
        dataset: BaseDataset = None,
        description="running prediction",
    ):
        """
        Predicts using a model against for a specified dataset

        :return: tuple of (y_true, y_pred, y_pred_probs)
        :rtype: tuple
        """
        # initialize counters
        y_true = []
        y_pred = []
        y_pred_probs = []

        # predict
        for inputs, outputs, labels in self.predict_output_per_batch(
            dataset.dataloader(), description
        ):
            predicted_probs, predicted = self.get_predicted(outputs)
            y_pred_probs += list(predicted_probs.cpu().detach().numpy())
            y_pred += list(predicted.cpu().detach().numpy())
            y_true += list(labels.cpu().detach().numpy())

        return y_true, y_pred, y_pred_probs

    def predict_image(
        self,
        image=None,
        labels=None,
        data_transforms=None,
        description="running prediction for single image",
    ):
        """
        Predicts using a model against for a specified image

        :return: Plot containing the image and the predictions.
        :rtype: matplotlib.figure.Figure
        """
        # load the image and apply transformations
        original_image = copy.deepcopy(image)
        self.model.eval()
        if data_transforms:
            image = data_transforms(image)
        # check if tensor and convert to batch of size 1, otherwise convert to tensor and then to batch of size 1
        if torch.is_tensor(image):
            inputs = image.unsqueeze(0).to(self.device)
        else:
            inputs = torch.from_numpy(image).unsqueeze(0).to(self.device)
        outputs = self(inputs)
        # check if outputs is OrderedDict for segmentation
        if isinstance(outputs, collections.abc.Mapping):
            outputs = outputs["out"]

        predicted_probs, predicted = self.get_predicted(outputs)
        y_pred_probs = list(predicted_probs.cpu().detach().numpy())

        """Display image and predictions from model"""
        # Convert results to dataframe for plotting
        result = pd.DataFrame({"p": y_pred_probs[0]}, index=labels)
        # Show the image
        plt.rcParams.update({"font.size": 16})
        fig = plt.figure(figsize=(16, 7))
        ax = plt.subplot(1, 2, 1)
        ax.axis("off")
        ax.imshow(original_image)

        # Set title to be the actual class
        ax.set_title("", size=20)

        ax = plt.subplot(1, 2, 2)
        # Plot a bar plot of predictions
        result.sort_values("p")["p"].plot.barh(color="blue", edgecolor="k", ax=ax)
        plt.xlabel("Predicted Probability")
        plt.tight_layout()

        return fig

    def predict_masks(
        self,
        image=None,
        labels=None,
        data_transforms=None,
        description="running prediction for single image",
    ):
        """
        Predicts using a model against for a specified image

        :return: Plot of the predicted masks
        :rtype: matplotlib.figure.Figure
        """
        # load the image and apply transformations
        original_image = copy.deepcopy(image)
        self.model.eval()
        if data_transforms:
            image = data_transforms(image)
        # check if tensor and convert to batch of size 1, otherwise convert to tensor and then to batch of size 1
        if torch.is_tensor(image):
            inputs = image.unsqueeze(0).to(self.device)
        else:
            inputs = torch.from_numpy(image).unsqueeze(0).to(self.device)
        outputs = self(inputs)
        # check if outputs is OrderedDict for segmentation
        if isinstance(outputs, collections.abc.Mapping):
            outputs = outputs["out"]

        predicted_probs, predicted = self.get_predicted(outputs)
        predicted_probs = list(predicted_probs.cpu().detach().numpy())
        predicted = list(predicted.cpu().detach().numpy())

        """Display image and masks from model"""
        # Show the image
        fig = plt.figure(figsize=(10, 10))

        # plot image
        plt.subplot(1, len(labels) + 1, 1)
        plt.imshow(original_image)
        plt.title("Image")
        plt.axis("off")

        # plot masks
        for i in range(len(labels)):
            plt.subplot(1, len(labels) + 1, i + 2)
            plt.imshow(
                predicted[0][i].astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255
            )
            plt.title(labels[i])
            plt.axis("off")

        plt.tight_layout()

        return fig

    def detect_objects(
        self,
        image=None,
        labels=None,
        data_transforms=None,
        description="running object detection for single image",
    ):
        """
        Predicts using a model against for a specified image

        :return: Plots the image with the object boundaries.
        :rtype: matplotlib.figure.Figure
        """
        # load the image and apply transformations
        image = image / 255
        self.model.eval()
        if data_transforms:
            image = data_transforms(image)
            original_image = copy.deepcopy(image)
            image = image.transpose(2, 0, 1)
        # check if tensor and convert to batch of size 1, otherwise convert to tensor and then to batch of size 1
        if torch.is_tensor(image):
            inputs = image.unsqueeze(0).to(self.device)
        else:
            inputs = (
                torch.from_numpy(image)
                .type(torch.FloatTensor)
                .unsqueeze(0)
                .to(self.device)
            )

        outputs = self(inputs)

        predicted = self.get_predicted(outputs)[0]
        """Display image and plot object boundaries"""
        fig, a = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)
        a.imshow(original_image)
        for box, label in zip(
            predicted["boxes"].cpu().detach().numpy(),
            predicted["labels"].cpu().detach().numpy(),
        ):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor="violet", facecolor="none"
            )

            # Draw the bounding box on top of the image
            a.add_patch(rect)
            a.annotate(
                labels[label],
                (box[0], box[1]),
                color="violet",
                fontsize=12,
                ha="center",
                va="center",
            )
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        plt.show()
        return fig

    def predict_output_per_batch(self, dataloader, description):
        """Run predictions on a dataloader and return inputs, outputs, labels per batch"""

        # turn on eval mode
        self.model.eval()

        # run predictions
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataloader, desc=description)):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(inputs)

                # check if outputs is OrderedDict for segmentation
                if isinstance(outputs, collections.abc.Mapping):
                    outputs = outputs["out"]

                yield inputs, outputs, labels

    def forward(self, *input, **kwargs):
        """
        Abstract method implementing the model. Extending classes should override this method.
        :return: Instance extending `nn.Module`
        :rtype: nn.Module
        """
        raise NotImplementedError

    def get_predicted(self, outputs, threshold=None):
        """Gets the output from the model and return the predictions
        :return: Tuple in the format (probabilities, predicted classes/labels)
        :rtype: tuple
        """
        raise NotImplementedError("Please implement `get_predicted` for your model. ")

    def report(self, labels, dataset_name, running_metrics, **kwargs):
        """The report we want to generate for the model"""
        return ()

    def log_metrics(self, output, labels, tag="train", writer=None, epoch=0):
        """Log the calculated metrics"""
        calculated_metrics = output
        logging.info(stringify(calculated_metrics))
        if writer:
            for cm in calculated_metrics:
                for key in cm:
                    metric = cm[key]
                    if isinstance(metric, list) or isinstance(metric, np.ndarray):
                        for i, sub in enumerate(metric):
                            writer.add_scalar(f"{key}/{labels[i]}/{tag}", sub, epoch)
                    else:
                        writer.add_scalar(f"{key}/{tag}", metric, epoch)

    def allocate_device(self, opts=None):
        """
        Put the model on CPU or GPU

        :return: Return the model on CPU or GPU.
        :rtype: nn.Module
        """
        self.model = self.model.to(self.device)
        if self.criterion:
            self.criterion = self.criterion.to(self.device)
        if self.config.use_ddp:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device]
            )
        return self.model

    def save_model(self, model_directory, epoch, optimizer, loss, start, run_id):
        """
        Saves the model on disk
        :param model_directory: directory to save the model
        :param epoch: Epoch number of checkpoint
        :param optimizer: Optimizer used
        :param loss: Criterion used
        :param start: Start time of training
        :param run_id: Run id of the model
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

    def extract_features(self, *input, **kwargs):
        """
        Abstract for trim the model to extract feature. Extending classes should override this method.

        :return: Instance of the model architecture
        :rtype: nn.Module
        """
        return self.model

    def load_model(self, file_path, optimizer=None):
        """Loads a model from a checkpoint"""
        if os.path.isfile(file_path):
            logging.info(f"Loading checkpoint {file_path}")
            checkpoint = torch.load(file_path)

            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)
                self.allocate_device()

                start_epoch = checkpoint["epoch"]
                loss = checkpoint["loss"]
                start = checkpoint["start"]
                run_id = checkpoint["id"]
            else:
                self.model.load_state_dict(checkpoint)
                self.allocate_device()

                start_epoch = 1
                loss = 0
                start = 0
                run_id = ""

            if optimizer:
                optimizer.load_state_dict(checkpoint["optimizer"])

            logging.info(f"Loaded checkpoint {file_path} at epoch {start_epoch}")
            return (start_epoch, loss, start, run_id)
        else:
            raise ValueError(f"No checkpoint found at {file_path}")

    def load_optimizer(self):
        """Load the optimizer"""
        raise NotImplementedError("Please implement `load_optimizer` for your model. ")

    def load_criterion(self):
        """Load the loss function"""
        raise NotImplementedError("Please implement `load_criterion` for your model. ")

    def load_lr_scheduler(self, optimizer):
        raise NotImplementedError(
            "Please implement `load_lr_scheduler` for your model. "
        )

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
        **kwargs,
    ):
        """Main method that trains the model.

        :param train_dataset: Dataset to train the model
        :type train_dataset: BaseDataset
        :param epochs: Number of epochs for training, defaults to 100
        :type epochs: int, optional
        :param model_directory: Directory where the model checkpoints will be saved, defaults to None
        :type model_directory: str, optional
        :param save_epochs: Number of epochs to save a checkpoint of the model, defaults to 10
        :type save_epochs: int, optional
        :param iterations_log: The number of iterations to pass before logging the system state, defaults to 100
        :type iterations_log: int, optional
        :param resume_model: Boolean indicating whether to resume an already traind model or not, defaults to None
        :type resume_model: str, optional
        :param val_dataset: Dataset used for validation, defaults to None
        :type val_dataset: BaseDataset, optional
        :param run_id: Optional run id to identify the experiment, defaults to None
        :type run_id: str, optional
        :return: Return the loss of the model
        """
        return self.fit(
            dataset=train_dataset,
            epochs=epochs,
            model_directory=model_directory,
            save_epochs=save_epochs,
            iterations_log=iterations_log,
            resume_model=resume_model,
            run_id=run_id,
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
        **kwargs,
    ):
        """Method that trains and evaluates the model.

        :param train_dataset: Dataset to train the model
        :type train_dataset: BaseDataset
        :param epochs: Number of epochs for training, defaults to 100
        :type epochs: int, optional
        :param model_directory: Model directory where the model checkpoints will be saved, defaults to None
        :type model_directory: str, optional
        :param save_epochs: Number of epochs to save a checkpoint of the model, defaults to 10
        :type save_epochs: int, optional
        :param iterations_log: Number of iterations to pass before logging the system state, defaults to 100
        :type iterations_log: int, optional
        :param resume_model: Boolean indicating whether to resume an already traind model or not, defaults to None
        :type resume_model: str, optional
        :param val_dataset: Dataset used for validation, defaults to None
        :type val_dataset: BaseDataset, optional
        :param run_id: Run id to identify the experiment, defaults to None
        :type run_id: str, optional
        :return: Loss of the model
        """
        return self.fit(
            dataset=train_dataset,
            epochs=epochs,
            model_directory=model_directory,
            save_epochs=save_epochs,
            iterations_log=iterations_log,
            resume_model=resume_model,
            val_dataset=val_dataset,
            run_id=run_id,
            **kwargs,
        )
