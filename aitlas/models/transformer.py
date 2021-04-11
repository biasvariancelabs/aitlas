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
from .schemas import TransformerModelSchema

#__all__ = ['TransformerModel']

class TransformerModel(BaseMulticlassClassifier):
    """Transformer Model for Multi-Class Classification"""

    schema = TransformerModelSchema

    def __init__(self, config):
        BaseMulticlassClassifier.__init__(self, config)

        #self.modelname = f"TransformerEncoder_input-dim={input_dim}_num-classes={num_classes}_" \
        #                 f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_" \
        #                 f"dropout={dropout}"

        encoder_layer = TransformerEncoderLayer(self.config.d_model, self.config.n_head, self.config.d_inner, self.config.dropout, self.config.activation)
        encoder_norm = LayerNorm(self.config.d_model)

        self.model.inlinear = Linear(self.config.input_dim, self.config.d_model)
        self.model.relu = ReLU()
        self.model.transformerencoder = TransformerEncoder(encoder_layer, self.config.n_layers, encoder_norm)
        self.model.flatten = Flatten()
        self.model.outlinear = Linear(self.config.d_model, self.config.num_classes)

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
