import csv
import json
import os

import numpy as np
import torch
from eolearn.core import EOPatch, FeatureType

from ..base import BaseDataset
from .schemas import EOPatchDatasetSchema


def eopatch_dataset(root_dir, features_data, fill_na=None):
    """Reads a folder of EOPatches from a specified directory"""

    def _read_patch(path):
        """Reading an eopatch at a given path. """

        def _func(path):
            # Load only relevant features
            features = [(data[0], data[1]) for data in features_data]
            patch = EOPatch.load(path, features=features)

            data = []
            for feat_type, feat_name, out_name, dtype, shape in features_data:
                arr = patch[feat_type][feat_name].astype(dtype)

                if fill_na is not None:
                    arr[np.isnan(arr)] = fill_na

                data.append(arr)

            return data

        out_types = [data[3] for data in features_data]
        # data = [x.astype(out_types) for x in _func(path)]
        data = [x for x in _func(path)]

        out_data = {}
        for f_data, feature in zip(features_data, data):
            feat_type, feat_name, out_name, dtype, shape = f_data
            # print(f_data)
            # feature.reshape(shape)
            out_data[out_name] = feature

        return out_data

    dataset = []
    dir = os.path.expanduser(root_dir)
    for patch in os.listdir(dir):
        dataset.append(_read_patch(os.path.join(dir, patch)))
    return dataset


class EOPatchDataset(BaseDataset):

    schema = EOPatchDatasetSchema

    classes_to_idx = {1: 0}

    def __init__(self, config):
        BaseDataset.__init__(self, config)
        print(self.config.root)

        features_data = [
            (
                self.config.input_feature_type,
                self.config.input_feature_name,
                "features",
                np.float32,
                self._parse_shape(self.config.input_feature_shape),
            ),
            (
                self.config.labels_feature_type,
                self.config.labels_feature_name,
                "labels",
                np.int64,
                self._parse_shape(self.config.labels_feature_shape),
            ),
        ]
        self.data = eopatch_dataset(self.config.root, features_data, fill_na=-2)

    def __getitem__(self, index):
        print(self.data[index])
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def labels(self):
        return [1]

    def _parse_shape(self, shape):
        shape = [None if s < 0 else s for s in shape]
        return shape
