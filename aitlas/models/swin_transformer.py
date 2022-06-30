import timm
import torch.nn as nn

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class SwinTransformer(BaseMulticlassClassifier):
    name = "SwinTransformerV2"

    def __init__(self, config):
        super().__init__(config)

        self.model = timm.create_model(
            "swinv2_cr_base_224", pretrained=self.config.pretrained
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

    def forward(self, x):
        return self.model(x)


class SwinTransformerMultilabel(BaseMultilabelClassifier):
    name = "SwinTransformerV2"

    def __init__(self, config):
        super().__init__(config)

        self.model = timm.create_model(
            "swinv2_cr_base_224", pretrained=self.config.pretrained
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

    def forward(self, x):
        return self.model(x)
