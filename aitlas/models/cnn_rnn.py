import collections
import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet152
from tqdm import tqdm

from aitlas.base import BaseModel, BaseDataset
from aitlas.models.schemas import CNNRNNModelSchema
from aitlas.utils import current_ts


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained Resnet-152 neural network and replace the top fc layer"""
        super(EncoderCNN, self).__init__()
        resnet = resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # ignore the last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images"""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        return self.bn(self.linear(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_classes, num_layers):
        """Set the hyperparameters and build the layers"""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, features, labels):
        """Decode image feature vectors and generate captions"""
        labels = labels.clone().detach().to(torch.int64)
        embeddings = self.embed(labels)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, torch.LongTensor(list(map(len, embeddings))), batch_first=True)
        hidden, (ht, _) = self.lstm(packed)
        return self.linear(ht[-1])


class CNNRNN(BaseModel):
    """Inspired by https://github.com/Lin-Zhipeng/CNN-RNN-A-Unified-Framework-for-Multi-label-Image-Classification"""
    schema = CNNRNNModelSchema

    def __init__(self, config):
        super(CNNRNN, self).__init__(config)
        self.model.encoder = EncoderCNN(embed_size=self.config["embed_size"]).to(self.device)
        self.model.decoder = DecoderRNN(embed_size=self.config["embed_size"],
                                        hidden_size=self.config["hidden_size"],
                                        num_classes=self.config["num_classes"],
                                        num_layers=self.config["num_layers"]).to(self.device)

    def forward(self, inputs):
        images, labels = inputs
        features = self.model.encoder(images)
        return self.model.decoder(features, labels)

    def load_criterion(self):
        return nn.BCEWithLogitsLoss(weight=self.weights)

    def load_optimizer(self):
        return torch.optim.Adam(params=list(self.model.decoder.parameters()), lr=self.config.learning_rate)

    def load_lr_scheduler(self):
        pass

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
        logging.info("Starting training.")

        start_epoch = 0
        start = current_ts()

        # load the model if needs to resume training
        if resume_model:
            start_epoch, loss, start, run_id = self.load_model(
                resume_model, self.optimizer
            )
        # allocate device
        self.allocate_device()
        # get data loaders
        train_loader = dataset.dataloader()
        val_loader = None
        if val_dataset:
            val_loader = val_dataset.dataloader()
        for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
            loss = self.train_epoch(
                epoch, train_loader, self.optimizer, self.criterion, iterations_log
            )
            if epoch % save_epochs == 0:
                self.save_model(
                    model_directory, epoch, self.optimizer, loss, start, run_id
                )
            # adjust learning rate if needed
            if self.lr_scheduler:
                self.lr_scheduler.step()
            # evaluate against the train set
            train_loss = self.evaluate_model(
                train_loader,
                criterion=self.criterion,
                description="testing on train set",
            )

            # evaluate against a validation set if there is one
            if val_loader:
                val_loss = self.evaluate_model(
                    val_loader,
                    criterion=self.criterion,
                    description="testing on validation set",
                )
                # self.log_metrics(
                #     self.running_metrics.get_scores(self.metrics),
                #     dataset.get_labels(),
                #     "val",
                #     self.writer,
                #     epoch + 1,
                # )
                self.running_metrics.reset()
        # save the model in the end
        self.save_model(model_directory, epochs, self.optimizer, loss, start, run_id)
        logging.info(f"finished training. training time: {current_ts() - start}")

    def get_predicted(self, outputs):
        predicted_probs = torch.sigmoid(outputs)
        predicted = predicted_probs >= self.config.threshold
        return predicted_probs, predicted

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

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self((inputs, labels))

            loss = criterion(outputs, labels if len(labels.shape) == 1 else labels.type(torch.float))
            loss.backward()

            # perform a single optimization step
            optimizer.step()

            # log statistics
            running_loss += loss.item() * inputs.size(0)
            total_loss += loss.item() * inputs.size(0)

            if i % iterations_log == iterations_log - 1:  # print every iterations_log mini-batches
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
        """Run predictions on a dataloader and return inputs, outputs, labels per batch"""

        # turn on eval mode
        self.model.eval()

        # run predictions
        with torch.no_grad():
            for _, data in enumerate(tqdm(dataloader, desc=description)):
                inputs, labels = data
                inputs = inputs.to(self.device)

                features = self.model.encoder(inputs)
                features = features.unsqueeze(1)
                hiddens, _ = self.model.decoder.lstm(features, None)
                outputs = self.model.decoder.linear(hiddens.squeeze(1))

                yield inputs, outputs, labels

    def evaluate_model(
            self, dataloader, criterion=None, description="testing on validation set",
    ):
        """
        Evaluates the current model against the specified dataloader for the specified metrics
        :param dataloader:
        :param metrics: list of metric keys to calculate
        :criterion: Criterion to calculate loss
        :description: What to show in the progress bar
        :return: tuple of (metrics, y_true, y_pred)
        """
        self.model.eval()

        # initialize loss if applicable
        total_loss = 0.0

        for inputs, outputs, labels in self.predict_output_per_batch(dataloader, description):
            if criterion:
                batch_loss = criterion(outputs, labels)
                total_loss += batch_loss.item() * inputs.size(0)

            _, predicted = self.get_predicted(outputs)
            # self.running_metrics.update(labels.type(torch.int64), predicted.type(torch.int64))

        if criterion:
            total_loss = total_loss / len(dataloader.dataset)

        return total_loss
