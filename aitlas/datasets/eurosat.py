import torchvision
import torchvision.transforms as transforms

from ..base import BaseDataset
from .schemas import SplitableDatasetSchema


class EurosatDataset(BaseDataset):
    schema = SplitableDatasetSchema

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

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.dataset.__len__()
