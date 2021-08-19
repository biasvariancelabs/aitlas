import torch.nn as nn
import torchvision.models as models

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class ResNet50(BaseMulticlassClassifier):
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
        self.freeze()

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.fc.parameters():
            param.require_grad = True

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model


class ResNet152(BaseMulticlassClassifier):
    def __init__(self, config):
        BaseMulticlassClassifier.__init__(self, config)

        if self.config.pretrained:
            self.model = models.resnet152(self.config.pretrained, False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        else:
            self.model = models.resnet152(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model


class ResNet50MultiLabel(BaseMultilabelClassifier):
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
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model


class ResNet152MultiLabel(BaseMultilabelClassifier):
    def __init__(self, config):
        BaseMultilabelClassifier.__init__(self, config)

        if self.config.pretrained:
            self.model = models.resnet152(self.config.pretrained, False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
        else:
            self.model = models.resnet152(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model



