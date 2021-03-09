"""

    Adapted from:
        https://github.com/dl4sits/BreizhCrops

    Original implementation of BreizhCrops dataset:
        https://github.com/dl4sits/BreizhCrops/blob/master/breizhcrops/datasets/breizhcrops.py

"""

import os
import zipfile
import tarfile
import urllib

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import h5py

from ..base import BaseDataset
from .schemas import BreizhCropsSchema


## from ..utils import download_file, unzip, untar
from .urls import CODESURL, CLASSMAPPINGURL, INDEX_FILE_URLs, FILESIZES, SHP_URLs, H5_URLs, RAW_CSV_URL

"""

TO DO ELENA: move the following to utils

"""

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path, overwrite=False):
    if url is None:
        raise ValueError("download_file: provided url is None!")

    if not os.path.exists(output_path) or overwrite:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"file exists in {output_path}. specify overwrite=True if intended")

def unzip(zipfile_path, target_dir):
    with zipfile.ZipFile(zipfile_path) as zip:
        for zip_info in zip.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, target_dir)

def untar(filepath):
    dirname = os.path.dirname(filepath)
    with tarfile.open(filepath, 'r:gz') as tar:
        #tar.extractall(path=dirname)
	#tar = tarfile.open(tar_file)
        for member in tar.getmembers():
            if member.isreg():  # skip if the TarInfo is not files
                member.name = os.path.basename(member.name) # remove the path by reset it
                tar.extract(member,dirname) # extract 




PADDING_VALUE = -1

BANDS = {
    "L1C": ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id'],
    "L2A": ['doa', 'id', 'code_cultu', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B8A', 'B11', 'B12', 'CLD', 'EDG', 'SAT']
}

SELECTED_BANDS = {
			"L1C": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 
						'QA10', 'QA20', 'QA60', 'doa'],
			"L2A": ['doa','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
					'CLD', 'EDG', 'SAT',]
}

"""

TO DO ELENA: move the following to transforms

"""

def get_default_transform(level):

    #padded_value = PADDING_VALUE
    sequencelength = 45

    bands = BANDS[level]
    if level == "L1C":
        selected_bands = ['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
    elif level == "L2A":
        selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    selected_band_idxs = np.array([bands.index(b) for b in selected_bands])

    def transform(x):
        #x = x[x[:, 0] != padded_value, :]  # remove padded values

        # choose selected bands
        x = x[:, selected_band_idxs] * 1e-4  # scale reflectances to 0-1

        # choose with replacement if sequencelength smaller als choose_t
        replace = False if x.shape[0] >= sequencelength else True
        idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
        idxs.sort()

        x = x[idxs]

        return torch.from_numpy(x).type(torch.FloatTensor)
    return transform

def get_default_target_transform():
    return lambda y: torch.tensor(y, dtype=torch.long)


class BreizhCropsDataset(BaseDataset):
    """BreizhCrops - a crop type classification dataset"""

    schema = BreizhCropsSchema

    def __init__(self, config):
        BaseDataset.__init__(self, config)

        #      (self,
        #          region,
        #          root="breizhcrops_dataset",
        #          year=2017, level="L1C",
        #          transform=None,
        #          target_transform=None,
        #          filter_length=0,
        #          verbose=False,
        #          load_timeseries=True,
        #          recompile_h5_from_csv=False,
        #          preload_ram=False)

        # :param region: dataset region. choose from "frh01", "frh02", "frh03", "frh04", "belle-ile"
        # :param root: where the data will be stored. defaults to `./breizhcrops_dataset`
        # :param year: year of the data. currently only `2017`
        # :param level: Sentinel 2 processing level. Either `L1C` (top of atmosphere) or `L2A` (bottom of atmosphere)
        # :param transform: a transformation function applied to the raw data before retrieving a sample. Can be used for featured extraction or data augmentaiton
        # :param target_transform: a transformation function applied to the label.
        # :param filter_length: time series shorter than `filter_length` will be ignored
        # :param bool verbose: verbosity flag
        # :param bool load_timeseries: if False, no time series data will be loaded. Only index file and class initialization. Used mostly for tests
        # :param bool recompile_h5_from_csv: downloads raw csv files and recompiles the h5 databases. Only required when dealing with new datasets
        # :param bool preload_ram: loads all time series data in RAM at initialization. Can speed up training if data is stored on HDD.

        """
        
        TO DO ELENA: move transforms to config arguments
        
        """

        transform=None
        target_transform=None

        if transform is None:
            self.transform = get_default_transform(config.level)
        if target_transform is None:
            self.target_transform = get_default_target_transform()

        """
        
        TO DO ELENA: figure out how to do splits of the dataset (for now load only one region)
        
        """

        assert config.year in [2017, 2018]
        assert config.level in ["L1C", "L2A"]
        assert config.region in ["frh01", "frh02", "frh03", "frh04", "belle-ile"]

        self.region = config.region.lower()
        self.bands = BANDS[config.level]

        self.verbose = config.verbose
        self.year = config.year
        self.level = config.level

        if config.verbose:
            print(f"Initializing BreizhCrops region {config.region}, year {config.year}, level {config.level}")

        self.root = config.root
        self.h5path, self.indexfile, self.codesfile, self.shapefile, self.classmapping, self.csvfolder = \
            self.build_folder_structure(self.root, self.year, self.level, self.region)

        self.load_classmapping(self.classmapping)
        #self.classes_to_idx = self.get_classes_to_ind(self.classmapping)
        print("Path "+self.h5path)
        if os.path.exists(self.h5path):
            h5_database_ok = os.path.getsize(self.h5path) == FILESIZES[config.year][config.level][config.region]
        else:
            h5_database_ok = False

        if not os.path.exists(self.indexfile):
            download_file(INDEX_FILE_URLs[config.year][config.level][config.region], self.indexfile)

        if not h5_database_ok and config.recompile_h5_from_csv and config.load_timeseries:
            self.download_csv_files()
            self.write_index()
            self.write_h5_database_from_csv(self.index)
        if not h5_database_ok and not config.recompile_h5_from_csv and config.load_timeseries:
            self.download_h5_database()

        self.index = pd.read_csv(self.indexfile, index_col=None)
        self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
        if config.verbose:
            print(f"kept {len(self.index)} time series references from applying class mapping")

        # filter zero-length time series
        if self.index.index.name != "idx":
            self.index = self.index.loc[self.index.sequencelength > config.filter_length].set_index("idx")

        self.maxseqlength = int(self.index["sequencelength"].max())

        if not os.path.exists(self.codesfile):
            download_file(CODESURL, self.codesfile)
        self.codes = pd.read_csv(self.codesfile, delimiter=";", index_col=0)

        if config.preload_ram:
            self.X_list = list()
            with h5py.File(self.h5path, "r") as dataset:
                for idx, row in tqdm(self.index.iterrows(), desc="loading data into RAM", total=len(self.index)):
                    self.X_list.append(np.array(dataset[(row.path)]))
        else:
            self.X_list = None

        self.index.rename(columns={"meanQA60": "meanCLD"}, inplace=True)
        
        if "classid" not in self.index.columns or "classname" not in self.index.columns or "region" not in self.index.columns:
            # drop fields that are not in the class mapping
            self.index = self.index.loc[self.index["CODE_CULTU"].isin(self.mapping.index)]
            self.index[["classid", "classname"]] = self.index["CODE_CULTU"].apply(lambda code: self.mapping.loc[code])
            self.index["region"] = self.region
            self.index.to_csv(self.indexfile)
        
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

        if self.X_list is None:
            with h5py.File(self.h5path, "r") as dataset:
                X = np.array(dataset[(row.path)])
        else:
            X = self.X_list[index]

        # translate CODE_CULTU to class id
        y = self.mapping.loc[row["CODE_CULTU"]].id

        #npad = self.maxseqlength - X.shape[0]
        #X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=PADDING_VALUE)

        if self.transform is not None:
            # print(self.transform)
            X = self.transform(X)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return X, y #, row.id

    """

    TO DO ELENA: possibly move the download etc. somewhere else

    """

    def download_csv_files(self):
        zipped_file = os.path.join(self.root, str(self.year), self.level, f"{self.region}.zip")
        download_file(RAW_CSV_URL[self.year][self.level][self.region], zipped_file)
        unzip(zipped_file, self.csvfolder)
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

    def download_h5_database(self):
        print(f"downloading {self.h5path}.tar.gz")
        download_file(H5_URLs[self.year][self.level][self.region], self.h5path + ".tar.gz", overwrite=True)
        print(f"extracting {self.h5path}.tar.gz to {self.h5path}")
        untar(self.h5path + ".tar.gz")
        print(f"removing {self.h5path}.tar.gz")
        os.remove(self.h5path + ".tar.gz")
        print(f"checking integrity by file size...")
        assert os.path.getsize(self.h5path) == FILESIZES[self.year][self.level][self.region]
        print("ok!")

    def write_h5_database_from_csv(self, index):
        with h5py.File(self.h5path, "w") as dataset:
            for idx, row in tqdm(index.iterrows(), total=len(index), desc=f"writing {self.h5path}"):                             
                X = self.load(os.path.join(self.root, row.path))
                dataset.create_dataset(row.path, data=X)

    def get_codes(self):
        return self.codes

    def load_classmapping(self, classmapping):
        if not os.path.exists(classmapping):
            if self.verbose:
                print(f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}")
            download_file(CLASSMAPPINGURL, classmapping)
        else:
            if self.verbose:
                print(f"found classmapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        if self.verbose:
            print(f"read {self.nclasses} classes from {classmapping}")

    def get_classes_to_ind(self, classmapping):
        if not os.path.exists(classmapping):
            if self.verbose:
                print(f"no classmapping found at {classmapping}, downloading from {CLASSMAPPINGURL}")
            download_file(CLASSMAPPINGURL, classmapping)
        else:
            if self.verbose:
                print(f"found classmapping at {classmapping}")

        mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        mapping = mapping[['id','classname']].groupby("id").first()
        mapping = mapping.set_index("classnamedf.to_dict('index')")
        return mapping.T.to_dict('records')[0]

    def load_raw(self, csv_file):
        """['B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
               'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""
        sample = pd.read_csv(os.path.join(self.csvfolder, os.path.basename(csv_file)), index_col=0).dropna()

        # convert datetime to int
        sample["doa"] = pd.to_datetime(sample["doa"]).astype(int)
        sample = sample.groupby(by="doa").first().reset_index()

        return sample

    def load(self, csv_file):
        sample = self.load_raw(csv_file)
        
        selected_bands = SELECTED_BANDS[self.level]
        X = np.array(sample[selected_bands].values)	
        if np.isnan(X).any():
            t_without_nans = np.isnan(X).sum(1) > 0
            X = X[~t_without_nans]

        return X

    def load_culturecode_and_id(self, csv_file):
        sample = self.load_raw(csv_file)
        X = np.array(sample.values)
	
        if self.level=="L1C":
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

    def write_index(self):
        csv_files = os.listdir(self.csvfolder)
        listcsv_statistics = list()
        i = 1

        for csv_file in tqdm(csv_files):
            if self.level == "L1C":
                cld_index = SELECTED_BANDS["L1C"].index("QA60")
            elif self.level == "L2A":
                cld_index = SELECTED_BANDS["L2A"].index("CLD")

            X = self.load(os.path.join(self.csvfolder, csv_file))
            culturecode, id = self.load_culturecode_and_id(os.path.join(self.csvfolder, csv_file))

            if culturecode is None or id is None:
                continue

            listcsv_statistics.append(
                dict(
                    meanQA60=np.mean(X[:, cld_index]),
                    id=id,
                    CODE_CULTU=culturecode,
                    path=os.path.join(self.csvfolder, f"{id}" + ".csv"),
                    idx=i,
                    sequencelength=len(X)
                )
            )
            i += 1

        self.index = pd.DataFrame(listcsv_statistics)
        self.index.to_csv(self.indexfile)

