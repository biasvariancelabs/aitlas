import torchvision.models as models

from ..base import BaseClassifier


class ResNet50(BaseClassifier):
    def __init__(self, config):
        BaseClassifier.__init__(self, config)

        self.model = models.resnet50(False, False, num_classes=self.config.num_classes)

    def forward(self, x):
        return self.model.forward(x)
