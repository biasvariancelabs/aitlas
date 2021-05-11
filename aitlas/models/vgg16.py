import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.models as models

from ..base import BaseMultilabelClassifier


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data)


def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class VGG16MultiLabel(BaseMultilabelClassifier):
    def __init__(self, config):
        BaseMultilabelClassifier.__init__(self, config)

        if self.config.pretrained:
            self.model = models.vgg16(self.config.pretrained, False)
            self.model.classifier = self.model.classifier[:-1]  # remove final layer
            self.model.classifier.add_module(
                "6", nn.Linear(4096, self.config.num_classes, bias=True)
            )

        else:
            self.model = models.vgg16(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)

    def load_criterion(self):
        return nn.BCEWithLogitsLoss()

    def load_optimizer(self):
        return optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )

    def extract_features(self):
        """ Remove final layers if we only need to extract features """
        self.model.classifier = self.model.classifier[:-3]

        return self.model
