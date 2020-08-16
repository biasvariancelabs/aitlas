import torchvision
import torchvision.transforms as transforms

from ..base import BaseDataset
from .schemas import CifarDatasetSchema


class CifarDataset(BaseDataset):
    schema = CifarDatasetSchema

    def __init__(self, config):
        BaseDataset.__init__(self, config)

        self.dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=self.config.train,
            download=self.config.download,
            transform=self.transforms,
        )

    def load_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.dataset.__len__()
