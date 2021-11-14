"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of Transformer model:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TransformerModel.py

"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import LayerNorm, Linear, ReLU
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from ..base import BaseMulticlassClassifier
from .schemas import TransformerModelSchema


class TransformerModel(BaseMulticlassClassifier):
    """Transformer Model for Multi-Class Classification"""

    schema = TransformerModelSchema

    def __init__(self, config):
        super().__init__(config)

        encoder_layer = TransformerEncoderLayer(
            self.config.d_model,
            self.config.n_head,
            self.config.d_inner,
            self.config.dropout,
            self.config.activation,
        )
        encoder_norm = LayerNorm(self.config.d_model)

        self.model.inlinear = Linear(self.config.input_dim, self.config.d_model)
        self.model.relu = ReLU()
        self.model.transformerencoder = TransformerEncoder(
            encoder_layer, self.config.n_layers, encoder_norm
        )
        self.model.flatten = Flatten()
        self.model.outlinear = Linear(self.config.d_model, self.config.num_classes)

    def forward(self, x):
        x = self.model.inlinear(x)
        x = self.model.relu(x)
        x = x.transpose(0, 1)  # N x T x D -> T x N x D
        x = self.model.transformerencoder(x)
        x = x.transpose(0, 1)  # T x N x D -> N x T x D
        x = x.max(1)[0]
        x = self.model.relu(x)
        logits = self.model.outlinear(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


class Flatten(nn.Module):
    """Flatten module"""

    def forward(self, input):
        return input.reshape(input.size(0), -1)
