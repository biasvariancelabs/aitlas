import torch.nn as nn
import torchvision.models as models

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class EfficientNetV2(BaseMulticlassClassifier):
    """EfficientNetV2 model implementation based on <https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_m.html#torchvision.models.efficientnet_v2_m>
    """

    name = "EfficientNetV2"

    def __init__(self, config):
        super().__init__(config)
        if self.config.pretrained:
            self.model = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1, progress=False
            )
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.config.num_classes)
        else:
            self.model = models.efficientnet_v2_m(
                weights=None, progress=False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)


class EfficientNetV2MultiLabel(BaseMultilabelClassifier):
    name = "EfficientNetV2"

    def __init__(self, config):
        super().__init__(config)
        if self.config.pretrained:
            self.model = models.efficientnet_v2_m(
                weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1, progress=False
            )
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.config.num_classes)
        else:
            self.model = models.efficientnet_v2_m(
                weights=None, progress=False, num_classes=self.config.num_classes
            )

    def forward(self, x):
        return self.model(x)
