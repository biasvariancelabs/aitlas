import torch.nn as nn
import torchvision.models as models

from ..base import BaseClassifier


class ResNet50(BaseClassifier):
    def __init__(self, config):
        BaseClassifier.__init__(self, config)

        self.model = models.resnet50(
            self.config.pretrained, False, num_classes=self.config.num_classes
        )
        if self.config.pretrained:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)

    def forward(self, x):
        return self.model.forward(x)
