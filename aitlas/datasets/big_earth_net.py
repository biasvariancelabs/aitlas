import csv
import json
import os
import lmdb
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import cv2
from PIL import Image

from itertools import compress
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from ..base import BaseDataset
from .schemas import BigEarthNetSchema
from ..utils import tiff_loader

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

DISPLAY_NAMES = {
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "Agriculture and vegetation",
    "Annual crops associated with permanent crops": "Crops",
    "Natural grassland and sparsely vegetated areas": "Grassland",
    "Moors, heathland and sclerophyllous vegetation": "Moors and heathland",
    "Road and rail networks and associated land": "Road and rail networks",
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


def loads_pickle(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


def dumps_pickle(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


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

    return bigearthnet_19_labels_multihot, original_labels_multihot


class BigEarthNetDataset(BaseDataset):
    """BigEarthNet dataset adaptation"""

    schema = BigEarthNetSchema
    name = "Big Earth Net"

    def __init__(self, config):
        BaseDataset.__init__(self, config)
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.data_dir = self.config.data_dir
        self.lmdb_path = self.config.lmdb_path
        self.version = self.config.version
        self.selection = self.config.selection

        if self.lmdb_path and not self.config.import_to_lmdb:
            self.db = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        if self.version == '19 labels':
            self.labels = LABELS['BigEarthNet-19_labels']
        else:
            self.labels = LABELS['original_labels']

        self.patches = self.load_patches()

    def __getitem__(self, index):

        patch_name = self.patches[index]

        with self.db.begin(write=False) as txn:
            byteflow = txn.get(patch_name.encode())
            bands10, bands20, _, multihots_19, multihots_43 = loads_pickle(byteflow)
            bands10 = bands10 / 2000 * 255.0
            bands10 = np.clip(bands10, 0, 255).astype(np.uint8)

            if self.version == '19 labels':
                multihots = multihots_19.astype(np.float32)
            else:
                multihots = multihots_43.astype(np.float32)

            if self.selection == 'rgb':
                bands10 = bands10.astype(np.float32)[:, :, [2, 1, 0]]
                if self.transform:
                    bands10 = self.transform(bands10)
                if self.target_transform:
                    multihots = self.target_transform(multihots)

                return bands10, multihots

            elif self.selection == 'all':
                # TODO interpolate/merge bands10 and bands20
                bands20 = interp_band(bands20)
                bands10 = bands10.astype(np.float32)
                bands20 = bands20.astype(np.float32)

                if self.transform:
                    bands10, bands20, bands60, multihots = self.transform(
                        (bands10, bands20)
                    )
                if self.target_transform:
                    multihots = self.target_transform(multihots)

                return bands10, bands20, multihots

    def __len__(self):
        return len(self.patches)

    def get_labels(self):
        return list(self.labels.keys())

    def load_patches(self):
        if self.lmdb_path:
            patch_names = []
            if self.config.csv_file:
                with open(self.config.csv_file, "r") as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        patch_names.append(row[0])
            return patch_names

    def get_item_name(self, index):
        return self.patches[index]

    def show_image(self, index):
        labels_list = list(compress(self.labels.keys(), self[index][1]))
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with labels:\n "
            f"{str(labels_list).strip('[]')}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def save_image(self, index):
        labels_list = list(compress(self.labels.keys(), self[index][1]))
        text_file = open("annotations_{index}.txt".format(index=index), "w")
        text_file.write(str(labels_list).strip('[]'))
        text_file.close()
        img = Image.fromarray(self[index][0].astype(np.uint8))
        img.save("image_{index}.jpg".format(index=index))

    def show_batch(self, size, show_title=True):
        if size % 3:
            raise ValueError("The provided size should be divided by 3!")
        image_indices = random.sample(range(0, len(self.patches)), size)
        figure, ax = plt.subplots(int(size / 3), 3, figsize=(13.75, 2.0 * int(size / 3)))
        if show_title:
            figure.suptitle(
                "Example images with labels from {}".format(self.get_name()),
                fontsize=32,
                y=1.006,
            )
        for axes, image_index in zip(ax.flatten(), image_indices):
            labels_list = list(compress(self.labels.keys(), self[image_index][1]))
            height, width, depth = self[image_index][0].shape
            white_image = np.zeros([height, 2*width, 3], dtype=np.uint8)
            white_image.fill(255)
            text = '\n'.join(labels_list)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.5
            font_thickness = 1
            x = 10

            for i, line in enumerate(text.split('\n')):
                if line in DISPLAY_NAMES.keys():
                    line = DISPLAY_NAMES[line]
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
                gap = textsize[1] + 5
                y = textsize[1] + i * gap
                cv2.putText(white_image, line, (x, y), font,
                            font_size,
                            (0, 0, 0),
                            font_thickness,
                            lineType=cv2.LINE_AA)

            display_image = np.hstack((self[image_index][0], white_image))
            axes.imshow(display_image)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axis('off')
        figure.tight_layout()
        return figure

    def data_distribution_table(self):
        distribution_table = {}
        for label in self.labels.keys():
            distribution_table[label] = 0

        for patch_index, patch_name in enumerate(self.patches):
            if patch_index and patch_index % 100000 == 0:
                print(f"Processed {patch_index} of {len(self.patches)}")

            _, multihots = self[patch_index]

            indices = [index for index, element in enumerate(multihots) if element == 1]
            for index in indices:
                key = [k for k, v in self.labels.items() if v == index]
                distribution_table[key[0]] += 1

        # creating a Dataframe object from a list of tuples of key, value pair
        label_count = pd.DataFrame(list(distribution_table.items()))
        label_count.columns = ["Label", "Count"]

        return label_count

    def data_distribution_barchart(self):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", data=label_count, ax=ax)
        ax.set_title("Image distribution for {}".format(self.get_name()), pad=20, fontsize=18)
        return fig

    def labels_stats(self):
        min_number = float('inf')
        max_number = float('-inf')
        average_number = 0
        for patch_index, patch_name in enumerate(self.patches):
            if patch_index and patch_index % 100000 == 0:
                print(f"Processed {patch_index} of {len(self.patches)}")

            _, multihots = self[patch_index]

            if sum(multihots) < min_number:
                min_number = sum(multihots)

            if sum(multihots) > max_number:
                max_number = sum(multihots)

            average_number += sum(multihots)

        return f"Minimum number of labels: {min_number}, Maximum number of labels: {max_number}, " \
               f"Average number of labels: {average_number / len(self.patches)}"

    def prepare(self):
        super().prepare()
        self.process_to_lmdb()

    def process_to_lmdb(self):
        patches = []
        dir = os.path.expanduser(self.data_dir)
        if os.path.isdir(dir):
            patches = sorted(os.listdir(dir))

        datagen = PrepBigEarthNetDataset(
            self.data_dir, patch_names_list=patches, label_indices=LABELS
        )
        dataloader = DataLoader(datagen, batch_size=1, num_workers=self.num_workers)

        patch_names = []
        self.db = lmdb.open(self.lmdb_path, map_size=1e12, readonly=False, meminit=False, map_async=True)
        txn = self.db.begin(write=True)
        for idx, data in enumerate(dataloader):
            print(f"Processed {idx} of {len(dataloader)}")
            bands10, bands20, bands60, patch_name, multihots_19, multihots_43 = data
            patch_name = patch_name[0]
            txn.put(
                u"{}".format(patch_name).encode("ascii"),
                dumps_pickle(
                    (
                        bands10[0].numpy(),
                        bands20[0].numpy(),
                        bands60[0].numpy(),
                        multihots_19[0].numpy(),
                        multihots_43[0].numpy()
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
            txn.put(b"__keys__", dumps_pickle(keys))
            txn.put(b"__len__", dumps_pickle(len(keys)))

        self.db.sync()
        self.db.close()


class PrepBigEarthNetDataset(Dataset):
    def __init__(self, data_dir=None, patch_names_list=None, label_indices=None):
        self.data_dir = data_dir
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
        patch_name = self.patch_names_list[idx]
        bands10_array = []
        bands20_array = []
        bands60_array = []

        for band in self.bands10:
            bands10_array.append(tiff_loader(
                os.path.join(self.data_dir, patch_name, patch_name + "_B" + band + ".tif")).astype(np.float32)
                                 )

        for band in self.bands20:
            bands20_array.append(tiff_loader(
                os.path.join(self.data_dir, patch_name, patch_name + "_B" + band + ".tif")).astype(np.float32)
                                 )

        for band in self.bands60:
            bands60_array.append(tiff_loader(
                os.path.join(self.data_dir, patch_name, patch_name + "_B" + band + ".tif")).astype(np.float32)
                                 )

        labels = parse_json_labels(
            os.path.join(self.data_dir, patch_name, patch_name + "_labels_metadata.json")
        )

        labels_multihot_19, labels_multihot_43 = cls2multihot(
            labels, self.label_indices
        )

        return (
            np.array(bands10_array).transpose(1, 2, 0),
            np.array(bands20_array).transpose(1, 2, 0),
            np.array(bands60_array).transpose(1, 2, 0),
            patch_name,
            np.array(labels_multihot_19),
            np.array(labels_multihot_43)
        )
