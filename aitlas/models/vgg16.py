import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.models as models

from ..base import BaseMultilabelClassifier
from .schemas import TorchModelSchema


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data)


def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)


class VGG16MultiLabel(BaseMultilabelClassifier):
    schema = TorchModelSchema

    def __init__(self, config):
        BaseMultilabelClassifier.__init__(self, config)

        if self.config.pretrained:
            self.model = models.vgg16(self.config.pretrained, False)

            self.model.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                *self.model.features[1:]
            )
            self.model.classifier = nn.Sequential(
                nn.Linear(4608, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, self.config.num_classes, bias=True),
            )

            self.model.apply(weights_init_kaiming)
            self.model.apply(fc_init_weights)
        else:
            self.model = models.vgg16(
                self.config.pretrained, False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        x = self.model.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return x

    def load_criterion(self):
        return nn.BCEWithLogitsLoss()

    def load_optimizer(self):
        return optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )
