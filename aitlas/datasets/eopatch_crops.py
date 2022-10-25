import logging
import os
import urllib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aitlas.datasets.crops_classification import CropsDataset
from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import VectorToRasterTask
from sklearn.model_selection import train_test_split
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, overwrite=False):
    if url is None:
        raise ValueError("download_file: provided url is None!")

    if not os.path.exists(output_path) or overwrite:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )
    else:
        logging.info(
            f"file exists in {output_path}. specify overwrite=True if intended"
        )


BANDS = ["B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12", "NDVI", "NDWI", "Brightness"]


class EOPatchCrops(CropsDataset):
    """EOPatchCrops - a crop type classification dataset"""

    def __init__(self, config):
        super().__init__(config)

        self.root = self.config.root
        self.regions = self.config.regions
        self.indexfile = os.path.join(self.config.root, self.config.csv_file_path)
        self.h5path = {}

        self.split_sets = ["train", "test", "val"]

        for region in self.split_sets:
            self.h5path[region] = os.path.join(self.config.root, f"{region}.hdf5")
        self.classmappingfile = os.path.join(self.config.root, "classmapping.csv")

        self.load_classmapping(self.classmappingfile)

        # Only do the timeseries (breizhcrops) file structure generation once, if a general index doesn't exist
        if not os.path.isfile(self.indexfile):
            self.preprocess()

        self.selected_bands = BANDS

        self.index = pd.read_csv(
            os.path.join(self.config.root, f"{self.regions[0]}.csv"), index_col=None
        )

        for region in self.regions[1:]:
            region_ind = pd.read_csv(
                os.path.join(self.config.root, f"{region}.csv"), index_col=None
            )
            self.index = pd.concat([self.index, region_ind], axis=0, ignore_index=True)

        self.X_list = None

    def preprocess(self):
        self.eopatches = [
            f.name
            for f in os.scandir(os.path.join(self.root, "eopatches"))
            if f.is_dir()
        ]
        self.indexfile = os.path.join(self.root, "index.csv")
        columns = [
            "path",
            "eopatch",
            "polygon_id",
            "CODE_CULTU",
            "sequencelength",
            "classid",
            "classname",
            "region",
        ]
        list_index = list()
        for patch in self.eopatches:
            eop = EOPatch.load(os.path.join(self.root, "eopatches", patch))
            polygons = eop.vector_timeless["CROP_TYPE_GDF"]
            for row in polygons.itertuples():
                if row.ct_eu_code not in self.mapping.index.values:
                    continue
                poly_id = int(row.polygon_id)

                classid = self.mapping.loc[row.ct_eu_code].id
                classname = self.mapping.loc[row.ct_eu_code].classname

                list_index.append(
                    {
                        columns[0]: os.path.join(patch, str(poly_id)),
                        columns[1]: patch,
                        columns[2]: poly_id,
                        columns[3]: row.ct_eu_code,
                        columns[4]: 0,
                        columns[5]: classid,
                        columns[6]: classname,
                        columns[7]: "",
                    }
                )
        self.index = pd.DataFrame(list_index)

        self.split()

        f = {}
        for set in self.split_sets:
            f[set] = h5py.File(self.h5path[set], "w")

        self.index.set_index("path", drop=False, inplace=True)

        for patch in self.eopatches:
            eop = EOPatch.load(os.path.join(self.root, "eopatches", patch))
            polygons = eop.vector_timeless["CROP_TYPE_GDF"]
            for row in polygons.itertuples():
                if row.ct_eu_code not in self.mapping.index.values:
                    continue
                poly_id = int(row.polygon_id)
                index_row = self.index.loc[os.path.join(patch, str(poly_id))]

                polygon = polygons[polygons.polygon_id == poly_id]
                temp = VectorToRasterTask(
                    vector_input=polygon,
                    raster_feature=(FeatureType.MASK_TIMELESS, "poly"),
                    values=1,
                    raster_shape=(FeatureType.MASK_TIMELESS, "CROP_TYPE"),
                )
                polygon_indicator_mask = temp.execute(eop).mask_timeless["poly"]

                seq_length = eop.data["FEATURES_S2"].shape[0]
                num_bands = eop.data["FEATURES_S2"].shape[3]

                polygon_indicator_mask_ts = np.repeat(
                    polygon_indicator_mask[np.newaxis, :, :, :], seq_length, axis=0
                )
                polygon_indicator_mask_ts = np.repeat(
                    polygon_indicator_mask_ts, num_bands, axis=3
                )

                temp_X = np.sum(
                    np.multiply(polygon_indicator_mask_ts, eop.data["FEATURES_S2"]),
                    axis=(1, 2),
                )

                dset = f[index_row.region].create_dataset(
                    patch + os.sep + str(poly_id), data=temp_X
                )
        self.index.reset_index(inplace=True, drop=True)
        self.write_index()

    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.index.values, self.index.classid.values, test_size=0.15, random_state=1
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=1
        )

        X_train = pd.DataFrame(X_train, columns=self.index.columns)
        X_train["region"] = "train"
        X_train.to_csv(os.path.join(self.root, "train.csv"))
        X_test = pd.DataFrame(X_test, columns=self.index.columns)
        X_test["region"] = "test"
        X_test.to_csv(os.path.join(self.root, "test.csv"))
        X_val = pd.DataFrame(X_val, columns=self.index.columns)
        X_val["region"] = "val"
        X_val.to_csv(os.path.join(self.root, "val.csv"))

        self.index = pd.concat([X_train, X_val, X_test], ignore_index=True)

    def write_index(self):
        self.index.to_csv(self.indexfile)
