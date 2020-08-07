import logging

import torch.nn as nn
import torch.nn.functional as F
from marshmallow.validate import ContainsOnly, OneOf

from ..base import BaseModel, Configurable
from .schemas import BaseClassifierSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Available losses. Add keys with new losses here.
losses = {"cross_entropy": nn.CrossEntropyLoss()}


# Available metrics. Add keys with new metrics here.
classification_metrics = {
    # 'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    # 'precision': tf.keras.metrics.Precision,
    # 'recall': tf.keras.metrics.Recall
}


class BaseClassifier(BaseModel):
    schema = BaseClassifierSchema

    def __init__(self, config):
        super().__init__(config)


class CifarModel(BaseModel):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
