"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of InceptionTime model:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/InceptionTime.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from ..base import BaseMulticlassClassifier
from .schemas import InceptionTimeSchema


class InceptionTime(BaseMulticlassClassifier):

    schema = InceptionTimeSchema

    def __init__(self, config):
        BaseMulticlassClassifier.__init__(self, config)

        self.model.inlinear = nn.Linear(
            self.config.input_dim, self.config.hidden_dims * 4
        )

        self.model.inception_modules_list = [
            InceptionModule(
                kernel_size=32,
                num_filters=self.config.hidden_dims * 4,
                use_bias=self.config.use_bias,
                device=self.device,
            )
            for _ in range(self.config.num_layers)
        ]

        self.inception_modules = nn.Sequential(*self.model.inception_modules_list)

        self.model.avgpool = nn.AdaptiveAvgPool1d(1)
        self.model.outlinear = nn.Linear(
            self.config.hidden_dims * 4, self.config.num_classes
        )

    def forward(self, x):
        # N x T x D -> N x D x T
        x = x.transpose(1, 2)

        # expand dimensions
        x = self.model.inlinear(x.transpose(1, 2)).transpose(1, 2)
        for i in range(self.config.num_layers):
            x = self.model.inception_modules_list[i](x)

        x = self.model.avgpool(x).squeeze(2)
        x = self.model.outlinear(x)
        logprobabilities = F.log_softmax(x, dim=-1)
        return logprobabilities

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


class InceptionModule(nn.Module):
    def __init__(
        self,
        kernel_size=32,
        num_filters=128,
        residual=True,
        use_bias=False,
        device=torch.device("cpu"),
    ):
        super(InceptionModule, self).__init__()

        self.residual = residual

        self.bottleneck = nn.Linear(num_filters, out_features=1, bias=use_bias)

        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convolutions = [
            nn.Conv1d(
                1,
                num_filters // 4,
                kernel_size=kernel_size + 1,
                stride=1,
                bias=use_bias,
                padding=kernel_size // 2,
            ).to(device)
            for kernel_size in kernel_size_s
        ]

        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(
                num_filters, num_filters // 4, kernel_size=1, padding=0, bias=use_bias
            ),
        )

        self.bn_relu = nn.Sequential(nn.BatchNorm1d(num_filters), nn.ReLU())

        if residual:
            self.residual_relu = nn.ReLU()

        # Maybe keep self.to here (it doesn't inherit from base)
        self.to(device)

    def forward(self, input_tensor):
        # collapse feature dimension
        input_inception = self.bottleneck(input_tensor.transpose(1, 2)).transpose(1, 2)
        features = [conv(input_inception) for conv in self.convolutions]
        features.append(self.pool_conv(input_tensor.contiguous()))
        features = torch.cat(features, dim=1)
        features = self.bn_relu(features)
        if self.residual:
            features = features + input_tensor
            features = self.residual_relu(features)
        return features
