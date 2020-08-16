import json
import unittest
from unittest import mock

from aitlas.base.datasets import SplitableDataset


class TestSplitableDataset(unittest.TestCase):
    def setUp(self) -> None:
        class SplitableDatasetFactory(SplitableDataset):
            def save_splits(self):
                return

            def __len__(self):
                return 10

            def __getitem__(self, item):
                return 1

        self.splitable_factory = SplitableDatasetFactory

    def test_read_empty(self):
        config = {
            "override": False,
            "split": {
                "train": {"ratio": 60, "file": "./data/train.csv"},
                "val": {"ratio": 20, "file": "./data/val.csv"},
                "test": {"ratio": 20, "file": "./data/test.csv"},
            },
        }

        try:
            obj = self.splitable_factory(config)
            assert False
        except ValueError:
            assert True

    def test_bad_split(self):
        config = {
            "override": True,
            "split": {
                "train": {"ratio": 70, "file": "./data/train.csv"},
                "val": {"ratio": 20, "file": "./data/val.csv"},
                "test": {"ratio": 20, "file": "./data/test.csv"},
            },
        }

        try:
            obj = self.splitable_factory(config)
            assert False
        except ValueError:
            assert True

    def test_split(self):
        config = {
            "override": True,
            "split": {
                "train": {"ratio": 60, "file": "./data/train.csv"},
                "val": {"ratio": 20, "file": "./data/val.csv"},
                "test": {"ratio": 20, "file": "./data/test.csv"},
            },
        }

        obj = self.splitable_factory(config)
        assert len(obj.train_set) == 6
        assert len(obj.test_set) == 2
        assert len(obj.val_set) == 2
