import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import numpy as np


from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier
from .schemas import ResnetSchema


class ResNet50(BaseMulticlassClassifier):
    schema = ResnetSchema

    def __init__(self, config):
        BaseMulticlassClassifier.__init__(self, config)

        if self.config.pretrained:
            self.model = models.resnet50(self.config.pretrained, False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        else:
            self.model = models.resnet50(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model.forward(x)


class ResNet50MultiLabel(BaseMultilabelClassifier):
    schema = ResnetSchema

    def __init__(self, config):
        BaseMultilabelClassifier.__init__(self, config)

        if self.config.pretrained:
            self.model = models.resnet50(self.config.pretrained, False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        else:
            self.model = models.resnet50(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model.forward(x)

    def load_criterion(self):
        """Load the loss function"""
        return nn.BCEWithLogitsLoss()

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)

    def get_predicted(self, outputs):
        predicted_probs = torch.sigmoid(outputs)
        predicted = predicted_probs >= 0.5
        return predicted_probs, predicted