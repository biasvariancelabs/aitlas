import os

from ..base import DatasetFolderMixin, SplitableDataset
from ..utils import pil_loader
from .schemas import RootFolderSchema


def generate_classes(dir):
    """Helper class to generate classes based on direct children folders of the dir"""
    dir = os.path.expanduser(dir)
    children_folders = [
        d for d in sorted(os.listdir(dir)) if os.path.isdir(os.path.join(dir, d))
    ]

    classes_to_index = {c: i for i, c in enumerate(children_folders)}

    return classes_to_index


class DataFolderDataset(SplitableDataset, DatasetFolderMixin):
    """Generic data"""

    schema = RootFolderSchema

    def __init__(self, config):
        # now call the constuctor to validate the schema and split the data
        SplitableDataset.__init__(self, config)

        self.classes_to_idx = generate_classes(self.config.root)

        self.data = self.make_dataset(self.config.root)

    def get_item_name(self, index):
        return self.data[index][0]

    def __getitem__(self, index):
        img = pil_loader(self.data[index][0])
        target = self.data[index][1]
        return img, target

    def __len__(self):
        return len(self.data)
