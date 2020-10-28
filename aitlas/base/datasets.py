import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import get_class
from .config import Configurable
from .schemas import BaseDatasetSchema
from .transforms import TRANSFORMS_PARAMS


class BaseDataset(Dataset, Configurable):

    schema = BaseDatasetSchema

    def __init__(self, config):
        Dataset.__init__(self)
        Configurable.__init__(self, config)

        # get dataloader parameters
        self.shuffle = self.config.shuffle
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers

        # get the transformations to be applied
        self.transform = self.load_transforms(self.config.transforms)

    def __getitem__(self, index):
        """ Implement here what you want to return"""
        raise NotImplementedError(
            "Please implement the `__getittem__` method for your dataset"
        )

    def __len__(self):
        raise NotImplementedError(
            "Please implement the `__len__` method for your dataset"
        )

    def prepare(self):
        """Implement if something needs to happen to the dataset after object creation"""
        return True

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def labels(self):
        """Implement this if you want to return the complete set of labels of the dataset"""
        raise NotImplementedError(
            "Please implement the `labels` method for your dataset"
        )

    def load_transforms(self, class_names):
        """Loads transformation classes and make a composition of them"""

        lst_transforms = []

        # check all transformation classes
        for name in class_names:
            cls = get_class(name)  # get class
            args = TRANSFORMS_PARAMS.get(name, None)  # get params, if specified
            if args:
                transfrm = cls(args)
            else:
                if cls.configurables:
                    kwargs = {}
                    for key in cls.configurables:
                        kwargs[key] = getattr(self.config, key)
                    transfrm = cls(**kwargs)
                else:
                    transfrm = cls()

            lst_transforms.append(transfrm)

        # return as composition
        return transforms.Compose(lst_transforms)
