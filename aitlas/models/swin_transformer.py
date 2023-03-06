import torchvision.models as models
import torch.nn as nn

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class SwinTransformer(BaseMulticlassClassifier):
    name = "SwinTransformerV2"

    def __init__(self, config):
        super().__init__(config)

        self.model = models.swin_v2_s(weights=
                                      models.Swin_V2_S_Weights.IMAGENET1K_V1
                                      if self.config.pretrained else None,
                                      progress=False)
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

        if self.config.freeze:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class SwinTransformerMultilabel(BaseMultilabelClassifier):
    name = "SwinTransformerV2"

    def __init__(self, config):
        super().__init__(config)

        self.model = models.swin_v2_s(weights=
                                      models.Swin_V2_S_Weights.IMAGENET1K_V1
                                      if self.config.pretrained else None,
                                      progress=False)
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

        if self.config.freeze:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
