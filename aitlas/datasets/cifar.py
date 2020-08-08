import torchvision
import torchvision.transforms as transforms

from ..base import BaseDataset, Configurable
from .schemas import CifarDatasetSchema


class CifarDataset(BaseDataset):
    schema = CifarDatasetSchema

    def __init__(self, config):
        BaseDataset.__init__(self, config)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=self.config.train,
            download=self.config.download,
            transform=transform,
        )
