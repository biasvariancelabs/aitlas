import torch.nn as nn
import torchvision.models as models

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class ConvNeXtTiny(BaseMulticlassClassifier):
    name = "ConvNeXt tiny"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.convnext_tiny(pretrained=self.config.pretrained)
            self.model.classifier = self.model.classifier[:-1]  # remove final layer
            self.model.classifier.add_module(
                "2", nn.Linear(768, self.config.num_classes, bias=True)
            )
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.convnext_tiny(
                pretrained=self.config.pretrained, progress=True, num_classes=self.config.num_classes
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
        self.model.classifier = self.model.classifier[:-1]

        return self.model


class ConvNeXtTinyMultiLabel(BaseMultilabelClassifier):
    name = "ConvNeXt tiny"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.convnext_tiny(pretrained=self.config.pretrained)
            self.model.classifier = self.model.classifier[:-1]  # remove final layer
            self.model.classifier.add_module(
                "2", nn.Linear(768, self.config.num_classes, bias=True)
            )
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.convnext_tiny(
                pretrained=self.config.pretrained, progress=True, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model.classifier = self.model.classifier[:-1]

        return self.model

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
