import torch.nn as nn
import torchvision.models as models

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class EfficientNetB0(BaseMulticlassClassifier):
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.efficientnet_b0(self.config.pretrained, False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.efficientnet_b0(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.classifier.parameters():
            param.require_grad = True

    """ Remove final layers if we only need to extract features """
    def extract_features(self):
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model


class EfficientNetB0MultiLabel(BaseMultilabelClassifier):
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.efficientnet_b0(self.config.pretrained, False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.efficientnet_b0(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model

    def freeze(self):
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.classifier.parameters():
            param.require_grad = True


class EfficientNetB4(BaseMulticlassClassifier):
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.efficientnet_b4(self.config.pretrained, False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.efficientnet_b4(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.classifier.parameters():
            param.require_grad = True

    """ Remove final layers if we only need to extract features """
    def extract_features(self):
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model


class EfficientNetB4MultiLabel(BaseMultilabelClassifier):
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.efficientnet_b4(self.config.pretrained, False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.efficientnet_b4(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model

    def freeze(self):
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.classifier.parameters():
            param.require_grad = True


class EfficientNetB7(BaseMulticlassClassifier):
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.efficientnet_b7(self.config.pretrained, False)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.efficientnet_b7(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.fc.parameters():
            param.require_grad = True

    """ Remove final layers if we only need to extract features """
    def extract_features(self):
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model


class EfficientNetB7MultiLabel(BaseMultilabelClassifier):
    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.efficientnet_b7(self.config.pretrained, False)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.efficientnet_b7(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model

    def freeze(self):
        for param in self.model.parameters():
            param.require_grad = False
        for param in self.model.classifier.parameters():
            param.require_grad = True



