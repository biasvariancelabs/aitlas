"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of OmniScaleCNN model:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/OmniScaleCNN.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..base import BaseMulticlassClassifier
from .schemas import OmniScaleCNNSchema


class SampaddingConv1D_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d(
            (int((kernel_size - 1) / 2), int(kernel_size / 2)), 0
        )
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        X = self.bn(X)
        return X


class build_layer_with_layer_parameter(nn.Module):
    """
    formerly build_layer_with_layer_parameter
    """

    def __init__(self, layer_parameters):
        """
        layer_parameters format
            [in_channels, out_channels, kernel_size,
            in_channels, out_channels, kernel_size,
            ..., nlayers
            ]
        """
        super(build_layer_with_layer_parameter, self).__init__()
        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            # in_channels, out_channels, kernel_size
            conv = SampaddingConv1D_BN(i[0], i[1], i[2])
            self.conv_list.append(conv)

    def forward(self, X):

        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(X)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result


class OmniScaleCNN(BaseMulticlassClassifier):
    """OmniScaleCNN Model for Multi-Class Classification"""

    schema = OmniScaleCNNSchema

    def __init__(self, config):
        super().__init__(config)
        receptive_field_shape = self.config.sequence_length // 4
        layer_parameter_list = generate_layer_parameter_list(
            1,
            receptive_field_shape,
            self.config.parameter_number_of_layer_list,
            in_channel=self.config.input_dim,
        )
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []

        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)

        self.model.net = nn.Sequential(*self.layer_list)

        self.model.averagepool = nn.AdaptiveAvgPool1d(1)

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr + final_layer_parameters[1]

        self.model.hidden = nn.Linear(out_put_channel_numebr, self.config.num_classes)

    def forward(self, X):
        X = self.model.net(X.transpose(1, 2))
        X = self.model.averagepool(X)
        X = X.squeeze_(-1)
        if not self.config.few_shot:
            X = self.model.hidden(X)
        return X

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1):
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer / (in_channel * sum(prime_list)))
    return out_channel_expect


def generate_layer_parameter_list(
    start, end, paramenter_number_of_layer_list, in_channel=1
):
    prime_list = get_Prime_number_in_a_range(start, end)

    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(
            paramenter_number_of_layer, in_channel, prime_list
        )

        tuples_in_layer = []
        for prime in prime_list:
            tuples_in_layer.append((in_channel, out_channel, prime))
        in_channel = len(prime_list) * out_channel

        layer_parameter_list.append(tuples_in_layer)

    tuples_in_layer_last = []
    first_out_channel = len(prime_list) * get_out_channel_number(
        paramenter_number_of_layer_list[0], 1, prime_list
    )
    tuples_in_layer_last.append((in_channel, first_out_channel, 1))
    tuples_in_layer_last.append((in_channel, first_out_channel, 2))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list
