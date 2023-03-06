import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class VGG16(BaseMulticlassClassifier):
    name = "VGG16"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.vgg16(self.config.pretrained, False)
            self.model.classifier = self.model.classifier[:-1]  # remove final layer
            self.model.classifier.add_module(
                "6", nn.Linear(4096, self.config.num_classes, bias=True)
            )
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.vgg16(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model.classifier = self.model.classifier[:-3]

        return self.model


class VGG19(BaseMulticlassClassifier):
    name = "VGG19"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.vgg19(self.config.pretrained, False)
            self.model.classifier = self.model.classifier[:-1]  # remove final layer
            self.model.classifier.add_module(
                "6", nn.Linear(4096, self.config.num_classes, bias=True)
            )
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.vgg19(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model.classifier = self.model.classifier[:-3]

        return self.model

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True


class VGG16MultiLabel(BaseMultilabelClassifier):
    name = "VGG16"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.vgg16(self.config.pretrained, False)
            self.model.classifier = self.model.classifier[:-1]  # remove final layer
            self.model.classifier.add_module(
                "6", nn.Linear(4096, self.config.num_classes, bias=True)
            )
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.vgg16(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model.classifier = self.model.classifier[:-3]

        return self.model

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True


class VGG19MultiLabel(BaseMultilabelClassifier):
    name = "VGG19"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.vgg19(self.config.pretrained, False)
            self.model.classifier = self.model.classifier[:-1]  # remove final layer
            self.model.classifier.add_module(
                "6", nn.Linear(4096, self.config.num_classes, bias=True)
            )
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.vgg19(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model.classifier = self.model.classifier[:-3]

        return self.model
