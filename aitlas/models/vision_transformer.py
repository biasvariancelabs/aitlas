import torch
import timm
import torch.nn as nn

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class VisionTransformer(BaseMulticlassClassifier):
    name = "ViT base_patch16_224"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained and self.config.local_model_path:
            checkpoint = torch.load(self.config.local_model_path)
            last_layer_key = next(reversed(checkpoint["state_dict"]))
            last_layer = checkpoint["state_dict"][last_layer_key]
            num_classes = len(last_layer)
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=False
            )
            self.model.head = nn.Linear(
                in_features=768, out_features=num_classes, bias=True
            )
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.model.head = nn.Linear(
                in_features=768, out_features=self.config.num_classes, bias=True
            )
        else:
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=self.config.pretrained
            )
            self.model.head = nn.Linear(
                in_features=768, out_features=self.config.num_classes, bias=True
            )

    def forward(self, x):
        return self.model(x)


class VisionTransformerMultilabel(BaseMultilabelClassifier):
    name = "ViT base_patch16_224"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained and self.config.local_model_path:
            checkpoint = torch.load(self.config.local_model_path)
            last_layer_key = next(reversed(checkpoint["state_dict"]))
            last_layer = checkpoint["state_dict"][last_layer_key]
            num_classes = len(last_layer)
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=False
            )
            self.model.head = nn.Linear(
                in_features=768, out_features=num_classes, bias=True
            )
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.model.head = nn.Linear(
                in_features=768, out_features=self.config.num_classes, bias=True
            )
        else:
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=self.config.pretrained
            )
            self.model.head = nn.Linear(
                in_features=768, out_features=self.config.num_classes, bias=True
            )

    def forward(self, x):
        return self.model(x)
