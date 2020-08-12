import torchvision
import torchvision.transforms as transforms

from ..base import SplitableDataset
from .schemas import SplitableDatasetSchema


class EurosatDataset(SplitableDataset):
    schema = SplitableDatasetSchema

    def __init__(self, config):
        SplitableDataset.__init__(self, config)

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 100
