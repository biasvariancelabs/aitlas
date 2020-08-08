import torch
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

    @property
    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
        )
