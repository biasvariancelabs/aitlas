"""MLP-Mixer architecture for image classification."""

import timm
import torch.nn as nn

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class MLPMixer(BaseMulticlassClassifier):
    """MLP mixer b16_224
    It creates an instance of the MLP mixer model version 'b16_224' from timm library (huggingface/pytorch-image-models).
    https://github.com/huggingface/pytorch-image-models for more details. """
    
    name = "MLP mixer b16_224"

    def __init__(self, config):
        super().__init__(config)

        self.model = timm.create_model(
            "mixer_b16_224", pretrained=self.config.pretrained
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

    def forward(self, x):
        return self.model(x)


class MLPMixerMultilabel(BaseMultilabelClassifier):
    name = "MLP mixer b16_224"

    def __init__(self, config):
        super().__init__(config)

        self.model = timm.create_model(
            "mixer_b16_224", pretrained=self.config.pretrained
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

    def forward(self, x):
        return self.model(x)
