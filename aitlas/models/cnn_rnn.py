import torch
import torch.nn as nn
from torchvision.models import resnet152

from ..base import BaseMultilabelClassifier
from .schemas import CNNRNNModelSchema


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # ignore the last fc layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        return self.bn(self.linear(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_classes, num_layers):
        super(DecoderRNN, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        features = features.unsqueeze(1)
        hiddens, _ = self.lstm(features, None)
        return self.linear(hiddens.squeeze(1))


class CNNRNN(BaseMultilabelClassifier):
    """Inspired by https://github.com/Lin-Zhipeng/CNN-RNN-A-Unified-Framework-for-Multi-label-Image-Classification"""

    schema = CNNRNNModelSchema

    def __init__(self, config):
        super(CNNRNN, self).__init__(config)
        self.model.encoder = EncoderCNN(embed_size=self.config["embed_size"]).to(
            self.device
        )
        self.model.decoder = DecoderRNN(
            embed_size=self.config["embed_size"],
            hidden_size=self.config["hidden_size"],
            num_classes=self.config["num_classes"],
            num_layers=self.config["num_layers"],
        ).to(self.device)

    def forward(self, inputs):
        return self.model.decoder(self.model.encoder(inputs))
