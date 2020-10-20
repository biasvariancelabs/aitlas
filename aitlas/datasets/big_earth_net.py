import json
import os
import csv

import lmdb
import numpy as np
import pyarrow as pa
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ..base import SplitableDataset, CsvDataset
from .schemas import BigEarthNetSchema, BigEarthNetRGBCsvSchema


LABELS = {
    "original_labels": {
        "Continuous urban fabric": 0,
        "Discontinuous urban fabric": 1,
        "Industrial or commercial units": 2,
        "Road and rail networks and associated land": 3,
        "Port areas": 4,
        "Airports": 5,
        "Mineral extraction sites": 6,
        "Dump sites": 7,
        "Construction sites": 8,
        "Green urban areas": 9,
        "Sport and leisure facilities": 10,
        "Non-irrigated arable land": 11,
        "Permanently irrigated land": 12,
        "Rice fields": 13,
        "Vineyards": 14,
        "Fruit trees and berry plantations": 15,
        "Olive groves": 16,
        "Pastures": 17,
        "Annual crops associated with permanent crops": 18,
        "Complex cultivation patterns": 19,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 20,
        "Agro-forestry areas": 21,
        "Broad-leaved forest": 22,
        "Coniferous forest": 23,
        "Mixed forest": 24,
        "Natural grassland": 25,
        "Moors and heathland": 26,
        "Sclerophyllous vegetation": 27,
        "Transitional woodland/shrub": 28,
        "Beaches, dunes, sands": 29,
        "Bare rock": 30,
        "Sparsely vegetated areas": 31,
        "Burnt areas": 32,
        "Inland marshes": 33,
        "Peatbogs": 34,
        "Salt marshes": 35,
        "Salines": 36,
        "Intertidal flats": 37,
        "Water courses": 38,
        "Water bodies": 39,
        "Coastal lagoons": 40,
        "Estuaries": 41,
        "Sea and ocean": 42,
    },
    "label_conversion": [
        [0, 1],
        [2],
        [11, 12, 13],
        [14, 15, 16, 18],
        [17],
        [19],
        [20],
        [21],
        [22],
        [23],
        [24],
        [25, 31],
        [26, 27],
        [28],
        [29],
        [33, 34],
        [35, 36],
        [38, 39],
        [40, 41, 42],
    ],
    "BigEarthNet-19_labels": {
        "Urban fabric": 0,
        "Industrial or commercial units": 1,
        "Arable land": 2,
        "Permanent crops": 3,
        "Pastures": 4,
        "Complex cultivation patterns": 5,
        "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
        "Agro-forestry areas": 7,
        "Broad-leaved forest": 8,
        "Coniferous forest": 9,
        "Mixed forest": 10,
        "Natural grassland and sparsely vegetated areas": 11,
        "Moors, heathland and sclerophyllous vegetation": 12,
        "Transitional woodland, shrub": 13,
        "Beaches, dunes, sands": 14,
        "Inland wetlands": 15,
        "Coastal wetlands": 16,
        "Inland waters": 17,
        "Marine waters": 18,
    },
}


def interp_band(bands, img10_shape=[120, 120]):
    """
    https://github.com/lanha/DSen2/blob/master/utils/patches.py
    """
    bands_interp = np.zeros([bands.shape[0]] + img10_shape).astype(np.float32)

    for i in range(bands.shape[0]):
        bands_interp[i] = resize(bands[i] / 30000, img10_shape, mode="reflect") * 30000

    return bands_interp


def parse_json_labels(f_j_path):
    """
    parse meta-data json file for big earth to get image labels
    :param f_j_path: json file path
    :return:
    """
    with open(f_j_path, "r") as f_j:
        j_f_c = json.load(f_j)
    return j_f_c["labels"]


def update_json_labels(f_j_path, BigEarthNet_19_labels):
    with open(f_j_path, "r") as f_j:
        j_f_c = json.load(f_j)

    j_f_c["BigEarthNet_19_labels"] = BigEarthNet_19_labels

    with open(f_j_path, "wb") as f:
        json.dump(j_f_c, f)


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def cls2multihot(cls_vec, label_indices):
    label_conversion = label_indices["label_conversion"]

    bigearthnet_19_label_idx = {
        v: k for k, v in label_indices["BigEarthNet-19_labels"].items()
    }

    bigearthnet_19_labels_multihot = np.zeros((len(label_conversion),))
    original_labels_multihot = np.zeros((len(label_indices["original_labels"]),))

    for cls_nm in cls_vec:
        original_labels_multihot[label_indices["original_labels"][cls_nm]] = 1

    for i in range(len(label_conversion)):
        bigearthnet_19_labels_multihot[i] = (
            np.sum(original_labels_multihot[label_conversion[i]]) > 0
        ).astype(int)

    bigearthnet_19_labels = []
    for i in np.where(bigearthnet_19_labels_multihot == 1)[0]:
        bigearthnet_19_labels.append(bigearthnet_19_label_idx[i])

    return bigearthnet_19_labels_multihot, bigearthnet_19_labels


def read_scale_raster(file_path, scale=1):
    """
    read raster file with specified scale
    :param file_path:
    :param scale:
    :return:
    """
    try:
        import gdal

        band_ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        return raster_band.ReadAsArray()
    except ImportError:
        try:
            import rasterio

            band_ds = rasterio.open(file_path)
            return np.array(band_ds.read(1))
        except ImportError:
            raise ImportError("You need to have `gdal` or `rasterio` installed. ")


class BaseBigEarthNetDataset(SplitableDataset):
    """BigEartNet dataset adaptation"""

    schema = BigEarthNetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        SplitableDataset.__init__(self, config)

        self.root = self.config.root
        self.num_workers = self.config.num_workers
        self.should_prepare = self.config.import_to_lmdb

        self.db = lmdb.open(config.lmdb_path)
        self.patches = self.load_patches(self.root)

    def __getitem__(self, index):
        patch_name = self.patches[index]
        print(patch_name)

        with self.db.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, _, _, multihots = loads_pyarrow(byteflow)

        bands10 = bands10.astype(np.float32)[0][
            0:3
        ]
        multihots = multihots.astype(np.float32)[0]

        if self.transform:
            bands10, multihots = self.transform(
                (bands10, multihots)
            )

        return bands10, multihots

    def __len__(self):
        return len(self.patches)

    def load_transforms(self):
        return transforms.Compose([
                        ToTensor(),
                        #Normalize(self.config.bands10_mean, self.config.bands10_std)
                    ])

    def load_patches(self, root):
        dir = os.path.expanduser(root)
        if os.path.isdir(dir):
            return sorted(os.listdir(dir))
        raise ValueError("`root` should be a folder")

    def get_item_name(self, index):
        return self.patches[index]

    def prepare(self):
        super().prepare()
        if self.should_prepare:
            self.process_to_lmdb()

    def process_to_lmdb(self):
        datagen = PrepBigEarthNetDataset(
            self.root, patch_names_list=self.patches, label_indices=LABELS
        )
        dataloader = DataLoader(datagen, num_workers=self.num_workers)

        patch_names = []
        txn = self.db.begin(write=True)
        for idx, data in enumerate(dataloader):
            bands10, bands20, bands60, patch_name, multihots = data
            patch_name = patch_name[0]
            txn.put(
                u"{}".format(patch_name).encode("ascii"),
                dumps_pyarrow(
                    (
                        bands10.numpy(),
                        bands20.numpy(),
                        bands60.numpy(),
                        multihots.numpy(),
                    )
                ),
            )
            patch_names.append(patch_name)

            if idx % 10000 == 0:
                txn.commit()
                txn = self.db.begin(write=True)

        txn.commit()
        keys = [u"{}".format(patch_name).encode("ascii") for patch_name in patch_names]

        with self.db.begin(write=True) as txn:
            txn.put(b"__keys__", dumps_pyarrow(keys))
            txn.put(b"__len__", dumps_pyarrow(len(keys)))

        self.db.sync()
        self.db.close()

class BigEarthNetRGBCsvDataset(CsvDataset):
    """BigEartNet dataset adaptation"""

    schema = BigEarthNetRGBCsvSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        CsvDataset.__init__(self, config)

        self.root = self.config.root
        self.num_workers = self.config.num_workers
        self.should_prepare = self.config.import_to_lmdb

        self.db = lmdb.open(config.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.patches = self.load_patches(self.root)

    def __getitem__(self, index):
        patch_name = self.patches[index]

        with self.db.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, _, _, multihots = loads_pyarrow(byteflow)

        bands10 = bands10.astype(np.float32)[0:3]
        multihots = multihots.astype(np.float32)

        if self.transform:
            bands10, multihots = self.transform(
                (bands10, multihots)
            )

        return bands10, multihots

    def __len__(self):
        return len(self.patches)

    def load_transforms(self):
        return transforms.Compose([
                        ToTensor(),
                        Normalize(self.config.bands10_mean, self.config.bands10_std)
                    ])

    def load_patches(self, root):
        patch_names = []
        if self.config.train_csv:
            with open(self.config.train_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    patch_names.append(row[0])

        if self.config.val_csv:
            with open(self.config.val_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    patch_names.append(row[0])

        if self.config.test_csv:
            with open(self.config.test_csv, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    patch_names.append(row[0])

        print(len(patch_names))

        return patch_names

    def get_item_name(self, index):
        return self.patches[index]

    def prepare(self):
        super().prepare()
        if self.should_prepare:
            self.process_to_lmdb()

    def process_to_lmdb(self):
        datagen = PrepBigEarthNetDataset(
            self.root, patch_names_list=self.patches, label_indices=LABELS
        )
        dataloader = DataLoader(datagen, num_workers=self.num_workers)

        patch_names = []
        txn = self.db.begin(write=True)
        for idx, data in enumerate(dataloader):
            bands10, bands20, bands60, patch_name, multihots = data
            patch_name = patch_name[0]
            txn.put(
                u"{}".format(patch_name).encode("ascii"),
                dumps_pyarrow(
                    (
                        bands10.numpy(),
                        bands20.numpy(),
                        bands60.numpy(),
                        multihots.numpy(),
                    )
                ),
            )
            patch_names.append(patch_name)

            if idx % 10000 == 0:
                txn.commit()
                txn = self.db.begin(write=True)

        txn.commit()
        keys = [u"{}".format(patch_name).encode("ascii") for patch_name in patch_names]

        with self.db.begin(write=True) as txn:
            txn.put(b"__keys__", dumps_pyarrow(keys))
            txn.put(b"__len__", dumps_pyarrow(len(keys)))

        self.db.sync()
        self.db.close()


class BigEarthNetRGBDataset(BaseBigEarthNetDataset):
    def __getitem__(self, index):
        patch_name = self.patches[index]

        with self.db.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, bands20, bands60, multihots = loads_pyarrow(byteflow)

        bands10 = bands10.astype(np.float32)[0][0:3]  # Return only RGB channels
        bands20 = bands20.astype(np.float32)[0]
        bands60 = bands60.astype(np.float32)[0]
        multihots = multihots.astype(np.float32)[0]

        if self.transform:
            bands10, bands20, bands60, multihots = self.transform(
                (bands10, bands20, bands60, multihots)
            )

        return bands10, multihots


class BigEarthNetAllBandsDataset(BaseBigEarthNetDataset):
    def __getitem__(self, index):
        patch_name = self.patches[index]

        with self.db.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())

        bands10, bands20, bands60, multihots = loads_pyarrow(byteflow)

        bands20 = interp_band(bands20)

        bands10 = bands10.astype(np.float32)[0]
        bands20 = bands20.astype(np.float32)[0]
        bands60 = bands60.astype(np.float32)[0]
        multihots = multihots.astype(np.float32)[0]

        if self.transform:
            bands10, bands20, bands60, multihots = self.transform(
                (bands10, bands20, bands60, multihots)
            )

        return bands10, multihots


class PrepBigEarthNetDataset(Dataset):
    def __init__(self, root=None, patch_names_list=None, label_indices=None):
        self.root = root
        self.label_indices = label_indices
        self.bands10 = ["02", "03", "04", "08"]
        self.bands20 = ["05", "06", "07", "8A", "11", "12"]
        self.bands60 = ["01", "09"]
        self.patch_names_list = patch_names_list

    def __len__(self):
        return len(self.patch_names_list)

    def __getitem__(self, index):
        return self.__data_generation(index)

    def __data_generation(self, idx):
        imgNm = self.patch_names_list[idx]

        bands10_array = []
        bands20_array = []
        bands60_array = []

        for band in self.bands10:
            bands10_array.append(
                read_scale_raster(
                    os.path.join(self.root, imgNm, imgNm + "_B" + band + ".tif")
                )
            )

        for band in self.bands20:
            bands20_array.append(
                read_scale_raster(
                    os.path.join(self.root, imgNm, imgNm + "_B" + band + ".tif")
                )
            )

        for band in self.bands60:
            bands60_array.append(
                read_scale_raster(
                    os.path.join(self.root, imgNm, imgNm + "_B" + band + ".tif")
                )
            )

        bands10_array = np.asarray(bands10_array).astype(np.float32)
        bands20_array = np.asarray(bands20_array).astype(np.float32)
        bands60_array = np.asarray(bands60_array).astype(np.float32)

        labels = parse_json_labels(
            os.path.join(self.root, imgNm, imgNm + "_labels_metadata.json")
        )
        BigEartNet_19_labels_multiHot, BigEarthNet_19_labels = cls2multihot(
            labels, self.label_indices
        )

        return (
            bands10_array,
            bands20_array,
            bands60_array,
            imgNm,
            BigEartNet_19_labels_multiHot,
        )


class Normalize(object):
    def __init__(self, bands_mean, bands_std):
        self.bands10_mean = bands_mean
        self.bands10_std = bands_std

    def __call__(self, sample):
        bands10, multihots = sample

        for t, m, s in zip(bands10, self.bands10_mean, self.bands10_std):
            t.sub_(m).div_(s)

        return bands10, multihots


class ToTensor(object):
    def __call__(self, sample):
        bands10, multihots = sample

        return torch.tensor(bands10), torch.tensor(bands20), bands60, multihots
