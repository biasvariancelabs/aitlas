import timm
import torch.nn as nn

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class VisionTransformer(BaseMulticlassClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=self.config.pretrained
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

    def forward(self, x):
        return self.model(x)


class VisionTransformerMultilabel(BaseMultilabelClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=self.config.pretrained
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

    def forward(self, x):
        return self.model(x)
