"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of BreizhCrops dataset:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/datasets/breizhcrops.py

"""
import logging
import os
import tarfile
import urllib
import zipfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aitlas.datasets.crops_classification import CropsDataset
from tqdm import tqdm

from ..base import BaseDataset
from .schemas import BreizhCropsSchema
from .urls import (
    CLASSMAPPINGURL,
    CODESURL,
    FILESIZES,
    RAW_CSV_URL,
    H5_URLs,
    INDEX_FILE_URLs,
    SHP_URLs,
)


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
            f"file exists in {output_path}. specify `overwrite=True` if intended"
        )


def unzip(zipfile_path, target_dir):
    with zipfile.ZipFile(zipfile_path) as zip:
        for zip_info in zip.infolist():
            if zip_info.filename[-1] == "/":
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, target_dir)


def untar(filepath):
    dirname = os.path.dirname(filepath)
    with tarfile.open(filepath, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isreg():  # skip if the TarInfo is not files
                member.name = os.path.basename(
                    member.name
                )  # remove the path by reset it
                tar.extract(member, dirname)  # extract


PADDING_VALUE = -1


BANDS = {
    "L1C": [
        "B1",
        "B10",
        "B11",
        "B12",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "QA10",
        "QA20",
        "QA60",
        "doa",
        "label",
        "id",
    ],
    "L2A": [
        "doa",
        "id",
        "code_cultu",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "CLD",
        "EDG",
        "SAT",
    ],
}

SELECTED_BANDS = {
    "L1C": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
        "QA10",
        "QA20",
        "QA60",
        "doa",
    ],
    "L2A": [
        "doa",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "CLD",
        "EDG",
        "SAT",
    ],
}


class BreizhCropsDataset(CropsDataset):
    """BreizhCrops - a crop type classification dataset"""

    schema = BreizhCropsSchema

    def __init__(self, config):
        super().__init__(config)
        # :param region: dataset region. choose from "frh01", "frh02", "frh03", "frh04", "belle-ile"
        # :param root: where the data will be stored. defaults to `./breizhcrops_dataset`
        # :param ф: year of the data. currently only `2017`
        # :param level: Sentinel 2 processing level. Either `L1C` (top of atmosphere) or `L2A` (bottom of atmosphere)
        # :param transform: a transformation function applied to the raw data before retrieving a sample. Can be used for featured extraction or data augmentaiton
        # :param target_transform: a transformation function applied to the label.
        # :param filter_length: time series shorter than `filter_length` will be ignored
        # :param bool verbose: verbosity flag
        # :param bool load_timeseries: if False, no time series data will be loaded. Only index file and class initialization. Used mostly for tests
        # :param bool recompile_h5_from_csv: downloads raw csv files and recompiles the h5 databases. Only required when dealing with new datasets
        # :param bool preload_ram: loads all time series data in RAM at initialization. Can speed up training if data is stored on HDD.

        self.regions = [region.lower() for region in self.config.regions]
        self.bands = BANDS[self.config.level]
        self.selected_bands = SELECTED_BANDS[self.config.level]
        self.root = self.config.root

        self.h5path = {}
        self.indexfile = {}
        self.shapefile = {}
        self.csvfolder = {}

        self.index = pd.DataFrame()

        self.preprocess()

    def preprocess(self):
        for region in self.regions:
            if self.config.verbose:
                logging.info(
                    f"Initializing BreizhCrops region {region}, year {self.config.year}, level {self.config.level}"
                )

            (
                self.h5path[region],
                self.indexfile[region],
                self.codesfile,
                self.shapefile[region],
                self.classmapping,
                self.csvfolder[region],
            ) = self.build_folder_structure(
                self.root, self.config.year, self.config.level, region
            )

            self.load_classmapping(self.classmapping)
            logging.info("Path " + self.h5path[region])
            if os.path.exists(self.h5path[region]):
                h5_database_ok = (
                    os.path.getsize(self.h5path[region])
                    == FILESIZES[self.config.year][self.config.level][region]
                )
            else:
                h5_database_ok = False

            if not os.path.exists(self.indexfile[region]):
                download_file(
                    INDEX_FILE_URLs[self.config.year][self.config.level][region],
                    self.indexfile[region],
                )

            if (
                not h5_database_ok
                and self.config.recompile_h5_from_csv
                and self.config.load_timeseries
            ):
                self.download_csv_files(region)
                self.write_index(region)
                self.write_h5_database_from_csv(self.index, region)
            if (
                not h5_database_ok
                and not self.config.recompile_h5_from_csv
                and self.config.load_timeseries
            ):
                self.download_h5_database(region)

            index_region = pd.read_csv(self.indexfile[region], index_col=None)
            index_region = index_region.loc[
                index_region["CODE_CULTU"].isin(self.mapping.index)
            ]

            if (
                "classid" not in index_region.columns
                or "classname" not in index_region.columns
                or "region" not in index_region.columns
            ):
                # drop fields that are not in the class mapping
                index_region = index_region.loc[
                    index_region["CODE_CULTU"].isin(self.mapping.index)
                ]
                index_region[["classid", "classname"]] = index_region[
                    "CODE_CULTU"
                ].apply(lambda code: self.mapping.loc[code])
                index_region["region"] = region
                index_region.to_csv(self.indexfile[region])

            if len(self.index.columns) == 0:
                self.index = pd.DataFrame(columns=index_region.columns)
            self.index = pd.concat(
                [self.index, index_region], axis=0, ignore_index=True
            )

        if self.config.verbose:
            logging.info(
                f"kept {len(self.index)} time series references from applying class mapping"
            )

        # filter zero-length time series
        if self.index.index.name != "idx":
            self.index = self.index.loc[
                self.index.sequencelength > self.config.filter_length
            ]  # set_index('idx')

        self.maxseqlength = int(self.index["sequencelength"].max())

        if not os.path.exists(self.codesfile):
            download_file(CODESURL, self.codesfile)
        self.codes = pd.read_csv(self.codesfile, delimiter=";", index_col=0)

        # for now
        self.X_list = None

        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)

        self.get_codes()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        row = self.index.iloc[index]

        h5path = self.h5path[row.region]

        if self.X_list is None:
            # Looks like this is what I need (load directly from file)
            with h5py.File(h5path, "r") as dataset:
                X = np.array(dataset[(row.path)])
        else:
            X = self.X_list[index]

        # translate CODE_CULTU to class id
        y = self.mapping.loc[row["CODE_CULTU"]].id

        if self.transform:
            X, y = self.transform((X, y))

        return X, y  # , row.id

    def get_labels(self):
        return self.index.classid

    # visualization functions

    def data_distribution_table(self):
        label_count = (
            self.index[["id", "region", "classname"]]
            .groupby(["classname", "region"])
            .count()
            .reset_index()
        )
        label_count.columns = ["Label", "Region", "Number of parcels"]
        return label_count

    def parcel_distribution_table(self):
        # Figure 2 a) in the paper
        parcel_count = (
            self.index[["id", "region"]].groupby("region").count().reset_index()
        )
        parcel_count.columns = ["Region NUTS-3", "# " + self.config.level]
        total_row = parcel_count.sum(axis=0)
        total_row["Region NUTS-3"] = "Total"

        parcel_count = parcel_count.append(total_row, ignore_index=True)
        return parcel_count

    def data_distribution_barchart(self):
        # Figure 2 b) in the paper
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        g = sns.barplot(
            x="Label", y="Number of parcels", hue="Region", data=label_count, ax=ax
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=30)
        g.set_yscale("log")
        return fig

    def show_samples(self):
        return self.index.head(20)

    def show_timeseries(self, index):
        # Figure 3 in the paper
        X, _ = self.__getitem__(index)
        label = row = self.index.iloc[index].loc["classname"]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(
            f"Timeseries with index {index} from the region {self.index.iloc[index].loc['region']}, with label {label}\n",
            fontsize=14,
        )
        ax.plot(X)
        ax.legend(BANDS[self.config.level][: X.shape[1]])
        ax.set_ylabel("ρ ")  # x ${10^4}$

        return fig

    """

    TO DO ELENA: possibly move the download etc. somewhere else

    """

    def download_csv_files(self, region):
        zipped_file = os.path.join(
            self.root, str(self.config.year), self.config.level, f"{region}.zip"
        )
        download_file(
            RAW_CSV_URL[self.config.year][self.config.level][region], zipped_file
        )
        unzip(zipped_file, self.csvfolder[region])
        os.remove(zipped_file)

    def build_folder_structure(self, root, year, level, region):
        """
        folder structure

        <root>
           codes.csv
           classmapping.csv
           <year>
              <region>.shp
              <level>
                 <region>.csv
                 <region>.h5
                 <region>
                     <csv>
                         123123.csv
                         123125.csv
                         ...
        """
        year = str(year)

        os.makedirs(os.path.join(root, year, level, region), exist_ok=True)

        h5path = os.path.join(root, year, level, f"{region}.h5")
        indexfile = os.path.join(root, year, level, region + ".csv")
        codesfile = os.path.join(root, "codes.csv")
        shapefile = os.path.join(root, year, f"{region}.shp")
        classmapping = os.path.join(root, "classmapping.csv")
        csvfolder = os.path.join(root, year, level, region, "csv")

        return h5path, indexfile, codesfile, shapefile, classmapping, csvfolder

    def get_fid(self, idx):
        return self.index[self.index["idx"] == idx].index[0]

    def download_h5_database(self, region):
        logging.info(f"downloading {self.h5path[region]}.tar.gz")
        download_file(
            H5_URLs[self.config.year][self.config.level][region],
            self.h5path[region] + ".tar.gz",
            overwrite=True,
        )
        logging.info(f"extracting {self.h5path[region]}.tar.gz to {self.h5path}")
        untar(self.h5path[region] + ".tar.gz")
        logging.info(f"removing {self.h5path[region]}.tar.gz")
        os.remove(self.h5path[region] + ".tar.gz")
        logging.info(f"checking integrity by file size...")
        assert (
            os.path.getsize(self.h5path[region])
            == FILESIZES[self.config.year][self.config.level][region]
        )
        logging.info("ok!")

    def write_h5_database_from_csv(self, index, region):
        with h5py.File(self.h5path[region], "w") as dataset:
            for idx, row in tqdm(
                index.iterrows(),
                total=len(index),
                desc=f"writing {self.h5path[region]}",
            ):
                X = self.load(os.path.join(self.root, row.path))
                dataset.create_dataset(row.path, data=X)

    def get_codes(self):
        return self.codes

    def load_classmapping(self, classmapping):
        if not os.path.exists(classmapping):
            if self.config.verbose:
                print(
                    f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}"
                )
            download_file(CLASSMAPPINGURL, classmapping)
        else:
            if self.config.verbose:
                print(f"found classmapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        if self.config.verbose:
            logging.info(f"read {self.nclasses} classes from {classmapping}")

    def get_classes_to_ind(self, classmapping):
        """ keep for now, could be needed to make it compatible with GenericMulticlass """
        if not os.path.exists(classmapping):
            if self.config.verbose:
                logging.info(
                    f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}"
                )
            download_file(CLASSMAPPINGURL, classmapping)
        else:
            if self.config.verbose:
                logging.info(f"found classmapping at {classmapping}")

        mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        mapping = mapping[["id", "classname"]].groupby("id").first()
        mapping = mapping.set_index("classnamedf.to_dict('index')")
        return mapping.T.to_dict("records")[0]

    def load_raw(self, csv_file):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
               'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""
        sample = pd.read_csv(
            os.path.join(self.csvfolder, os.path.basename(csv_file)), index_col=0
        ).dropna()

        # convert datetime to int
        sample["doa"] = pd.to_datetime(sample["doa"]).astype(int)
        sample = sample.groupby(by="doa").first().reset_index()

        return sample

    def load(self, csv_file):
        sample = self.load_raw(csv_file)

        selected_bands = SELECTED_BANDS[self.config.level]
        X = np.array(sample[selected_bands].values)
        if np.isnan(X).any():
            t_without_nans = np.isnan(X).sum(1) > 0
            X = X[~t_without_nans]

        return X

    def load_culturecode_and_id(self, csv_file):
        sample = self.load_raw(csv_file)
        X = np.array(sample.values)

        if self.config.level == "L1C":
            cc_index = self.bands.index("label")
        else:
            cc_index = self.bands.index("code_cultu")
        id_index = self.bands.index("id")

        if len(X) > 0:
            field_id = X[0, id_index]
            culture_code = X[0, cc_index]
            return culture_code, field_id

        else:
            return None, None

    def write_index(self, region):
        csv_files = os.listdir(self.csvfolder[region])
        listcsv_statistics = list()
        i = 1

        for csv_file in tqdm(csv_files):
            if self.config.level == "L1C":
                cld_index = SELECTED_BANDS["L1C"].index("QA60")
            elif self.config.level == "L2A":
                cld_index = SELECTED_BANDS["L2A"].index("CLD")

            X = self.load(os.path.join(self.csvfolder, csv_file))
            culturecode, id = self.load_culturecode_and_id(
                os.path.join(self.csvfolder, csv_file)
            )

            if culturecode is None or id is None:
                continue

            listcsv_statistics.append(
                dict(
                    meanQA60=np.mean(X[:, cld_index]),
                    id=id,
                    CODE_CULTU=culturecode,
                    path=os.path.join(self.csvfolder[region], f"{id}" + ".csv"),
                    idx=i,
                    sequencelength=len(X),
                )
            )
            i += 1

        self.index = pd.DataFrame(listcsv_statistics)
        self.index.to_csv(self.indexfile)
