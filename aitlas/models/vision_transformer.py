import torch
import timm
import torch.nn as nn
import logging

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class VisionTransformer(BaseMulticlassClassifier):
    name = "ViT base_patch16_224"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained and self.config.local_model_path:
            checkpoint = torch.load(self.config.local_model_path)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if "student" in checkpoint:
                checkpoint = checkpoint["student"]

            last_layer_key = next(reversed(checkpoint))
            last_layer = checkpoint[last_layer_key]
            num_classes = len(last_layer)
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=False
            )
            self.model.head = nn.Linear(
                in_features=768, out_features=num_classes, bias=True
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

        if self.config.freeze:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class VisionTransformerMultilabel(BaseMultilabelClassifier):
    name = "ViT base_patch16_224"

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained and self.config.local_model_path:
            checkpoint = torch.load(self.config.local_model_path)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if "student" in checkpoint:
                checkpoint = checkpoint["student"]

            last_layer_key = next(reversed(checkpoint))
            last_layer = checkpoint[last_layer_key]
            num_classes = len(last_layer)
            self.model = timm.create_model(
                "vit_base_patch16_224", pretrained=False
            )
            self.model.head = nn.Linear(
                in_features=768, out_features=num_classes, bias=True
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

        if self.config.freeze:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
