import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class ShallowCNNNet(BaseMulticlassClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model.conv1 = nn.Conv2d(3, 6, 5)
        self.model.pool = nn.MaxPool2d(2, 2)
        self.model.conv2 = nn.Conv2d(6, 16, 5)
        self.model.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.model.fc2 = nn.Linear(120, 84)
        self.model.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.model.pool(F.relu(self.model.conv1(x)))
        x = self.model.pool(F.relu(self.model.conv2(x)))

        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.model.fc1(x))
        x = F.relu(self.model.fc2(x))
        x = self.model.fc3(x)
        return x


class ShallowCNNNetMultilabel(BaseMultilabelClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model.conv1 = nn.Conv2d(3, 6, 5)
        self.model.pool = nn.MaxPool2d(2, 2)
        self.model.conv2 = nn.Conv2d(6, 16, 5)
        self.model.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.model.fc2 = nn.Linear(120, 84)
        self.model.fc3 = nn.Linear(84, self.config.num_classes)

    def forward(self, x):
        x = self.model.pool(F.relu(self.model.conv1(x)))
        x = self.model.pool(F.relu(self.model.conv2(x)))

        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.model.fc1(x))
        x = F.relu(self.model.fc2(x))
        x = self.model.fc3(x)
        return x
