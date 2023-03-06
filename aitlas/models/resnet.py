import torch
import torch.nn as nn
import torchvision.models as models
import logging

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ResNet50(BaseMulticlassClassifier):
    name = "ResNet50"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            if self.config.local_model_path:
                checkpoint = torch.load(self.config.local_model_path)

                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                if "student" in checkpoint:
                    checkpoint = checkpoint["student"]

                last_layer_key = next(reversed(checkpoint))
                last_layer = checkpoint[last_layer_key]
                num_classes = len(last_layer)
                self.model = models.resnet50(
                    weights=None, progress=False, num_classes=num_classes
                )
                # remove prefix "module."
                checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
                for k, v in self.model.state_dict().items():
                    if k not in list(checkpoint):
                        logging.info('key "{}" could not be found in provided state dict'.format(k))
                    elif checkpoint[k].shape != v.shape:
                        logging.info('key "{}" is of different shape in model and provided state dict'.format(k))
                        checkpoint[k] = v
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1, progress=False)

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.resnet50(
                weights=None, progress=False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model


class ResNet152(BaseMulticlassClassifier):
    name = "ResNet152"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.resnet152(self.config.pretrained, False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
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

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True


class ResNet50MultiLabel(BaseMultilabelClassifier):
    name = "ResNet50"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            if self.config.local_model_path:
                checkpoint = torch.load(self.config.local_model_path)
                last_layer_key = next(reversed(checkpoint["state_dict"]))
                last_layer = checkpoint["state_dict"][last_layer_key]
                num_classes = len(last_layer)
                self.model = models.resnet50(
                    weights=None, progress=False, num_classes=num_classes
                )
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1, progress=False)

            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
        else:
            self.model = models.resnet50(
                weights=None, progress=False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        return self.model

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True


class ResNet152MultiLabel(BaseMultilabelClassifier):
    name = "ResNet152"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = models.resnet152(self.config.pretrained, False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.config.num_classes)
            if self.config.freeze:
                self.freeze()
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

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
