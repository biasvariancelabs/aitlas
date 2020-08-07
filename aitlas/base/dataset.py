from torch.utils.data import Dataset

from .config import Configurable


class BaseDataset(Dataset, Configurable):
    def __init__(self, config):
        Dataset.__init__(self)
        Configurable.__init__(self, config)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
