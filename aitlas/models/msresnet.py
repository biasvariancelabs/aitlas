"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of MSResNet model:
        https://github.com/geekfeiw/Multi-Scale-1D-ResNet/blob/master/model/multi_scale_ori.py
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/models/MSResNet.py

"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torch.optim as optim

from ..base import BaseMulticlassClassifier
from .schemas import MSResNetSchema


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False
    )


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False
    )


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class MSResNet(BaseMulticlassClassifier):

    schema = MSResNetSchema

    def __init__(self, config):
        super().__init__(config)

        self.inplanes3 = self.config.hidden_dims
        self.inplanes5 = self.config.hidden_dims
        self.inplanes7 = self.config.hidden_dims
        stride = 2

        self.model.conv1 = nn.Conv1d(
            self.config.input_dim,
            self.config.hidden_dims,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.model.bn1 = nn.BatchNorm1d(self.config.hidden_dims)
        self.model.relu = nn.ReLU(inplace=True)
        self.model.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.model.layer3x3_1 = self._make_layer3(
            BasicBlock3x3, self.config.hidden_dims, self.config.layers[0], stride=stride
        )
        self.model.layer3x3_2 = self._make_layer3(
            BasicBlock3x3,
            2 * self.config.hidden_dims,
            self.config.layers[1],
            stride=stride,
        )
        self.model.layer3x3_3 = self._make_layer3(
            BasicBlock3x3,
            4 * self.config.hidden_dims,
            self.config.layers[2],
            stride=stride,
        )

        # maxplooing kernel size: 16, 11, 6
        self.model.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        self.model.layer5x5_1 = self._make_layer5(
            BasicBlock5x5, self.config.hidden_dims, self.config.layers[0], stride=stride
        )
        self.model.layer5x5_2 = self._make_layer5(
            BasicBlock5x5,
            2 * self.config.hidden_dims,
            self.config.layers[1],
            stride=stride,
        )
        self.model.layer5x5_3 = self._make_layer5(
            BasicBlock5x5,
            4 * self.config.hidden_dims,
            self.config.layers[2],
            stride=stride,
        )
        self.model.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.model.layer7x7_1 = self._make_layer7(
            BasicBlock7x7, self.config.hidden_dims, self.config.layers[0], stride=2
        )
        self.model.layer7x7_2 = self._make_layer7(
            BasicBlock7x7, 2 * self.config.hidden_dims, self.config.layers[1], stride=2
        )
        self.model.layer7x7_3 = self._make_layer7(
            BasicBlock7x7, 4 * self.config.hidden_dims, self.config.layers[2], stride=2
        )
        self.model.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        self.model.fc = nn.Linear(
            4 * self.config.hidden_dims * 3, self.config.num_classes
        )

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes3,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes5,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes7,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def _logits(self, x0):
        # require NxTxD format
        x0 = x0.transpose(1, 2)
        x0 = torch.nn.functional.interpolate(x0, size=512)

        x0 = self.model.conv1(x0)
        x0 = self.model.bn1(x0)
        x0 = self.model.relu(x0)
        x0 = self.model.maxpool(x0)

        x = self.model.layer3x3_1(x0)
        x = self.model.layer3x3_2(x)
        x = self.model.layer3x3_3(x)
        x = self.model.maxpool3(x)

        y = self.model.layer5x5_1(x0)
        y = self.model.layer5x5_2(y)
        y = self.model.layer5x5_3(y)
        y = self.model.maxpool5(y)

        z = self.model.layer7x7_1(x0)
        z = self.model.layer7x7_2(z)
        z = self.model.layer7x7_3(z)
        z = self.model.maxpool7(z)

        out = torch.cat([x, y, z], dim=1)

        out = out.squeeze()
        out1 = self.model.fc(out)

        return out1, out

    def forward(self, x0):
        logits, _ = self._logits(x0)

        logprobabilities = Functional.log_softmax(logits, dim=-1)

        return logprobabilities

    def load_optimizer(self):
        """Load the optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
