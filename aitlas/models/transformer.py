"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of Transformer model:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/TransformerModel.py

"""

from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, ReLU

import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseMulticlassClassifier

#__all__ = ['TransformerModel']

class TransformerModel(BaseMulticlassClassifier):
    """Transformer Model for Multi-Class Classification"""

    def __init__(self, config):
        BaseMulticlassClassifier.__init__(self, config)

        input_dim=13
        num_classes=config.num_classes
        d_model=64
        n_head=2
        n_layers=5
        d_inner=128
        activation="relu"
        dropout=0.017998950510888446
        self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
                         f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = LayerNorm(d_model)

        self.model.inlinear = Linear(input_dim, d_model)
        self.model.relu = ReLU()
        self.model.transformerencoder = TransformerEncoder(encoder_layer, n_layers, encoder_norm)
        self.model.flatten = Flatten()
        self.model.outlinear = Linear(d_model, num_classes)

    def forward(self,x):
        x = self.model.inlinear(x)
        x = self.model.relu(x)
        x = x.transpose(0, 1) # N x T x D -> T x N x D
        x = self.model.transformerencoder(x)
        x = x.transpose(0, 1) # T x N x D -> N x T x D
        x = x.max(1)[0]
        x = self.model.relu(x)
        logits = self.model.outlinear(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        return logprobabilities

class Flatten(nn.Module):
    """Flatten module"""

    def forward(self, input):
        return input.reshape(input.size(0), -1)
