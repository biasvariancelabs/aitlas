import torch.nn as nn
import torchvision.models as models

from ..base import BaseMulticlassClassifier
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
