"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of LSTM model:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/LongShortTermMemory.py

"""

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..base import BaseMulticlassClassifier
from .schemas import LSTMSchema


class LSTM(BaseMulticlassClassifier):
    """LSTM Model for Multi-Class Classification"""

    schema = LSTMSchema

    def __init__(self, config):
        super().__init__(config)

        if self.config.use_layernorm:
            self.model.inlayernorm = nn.LayerNorm(self.config.input_dim)
            self.model.clayernorm = nn.LayerNorm(
                (
                    self.config.hidden_dims
                    + self.config.hidden_dims * self.config.bidirectional
                )
                * self.config.num_layers
            )

        self.model.lstm = nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dims,
            num_layers=self.config.num_layers,
            bias=False,
            batch_first=True,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
        )

        if self.config.bidirectional:
            hidden_dims = self.config.hidden_dims * 2
        else:
            hidden_dims = self.config.hidden_dims

        self.model.linear_class = nn.Linear(
            hidden_dims * self.config.num_layers, self.config.num_classes, bias=True
        )

    def logits(self, x):
        if self.config.use_layernorm:
            x = self.model.inlayernorm(x)

        outputs, last_state_list = self.model.lstm.forward(x)

        h, c = last_state_list

        nlayers, batchsize, n_hidden = c.shape
        h = self.model.clayernorm(
            c.transpose(0, 1).contiguous().view(batchsize, nlayers * n_hidden)
        )
        logits = self.model.linear_class.forward(h)

        return logits

    def forward(self, x):
        logprobabilities = F.log_softmax(self.logits(x), dim=-1)
        return logprobabilities

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
