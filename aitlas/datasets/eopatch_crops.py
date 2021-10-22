
from aitlas.datasets.crops_classification import CropsDataset
import os
import zipfile
import tarfile
import urllib

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sns

import h5py

from ..base import BaseDataset

from .urls import CODESURL, CLASSMAPPINGURL, INDEX_FILE_URLs, FILESIZES, SHP_URLs, H5_URLs, RAW_CSV_URL

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import VectorToRasterTask

import matplotlib.pyplot as plt


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


BANDS = ['B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'Brightness']


class EOPatchCrops(CropsDataset):
    """EOPatchCrops - a crop type classification dataset"""

    def __init__(self, config):
        CropsDataset.__init__(self, config)
        
        self.root = self.config.root
        self.regions = self.config.regions #'slovenia'
        self.indexfile = self.config.root+os.sep+self.config.csv_file_path
        self.h5path = {}

        self.split_sets = ['train','test','val']

        for region in self.split_sets:
            self.h5path[region] = self.config.root+os.sep+region+'.hdf5'
        self.classmappingfile = self.config.root+os.sep+"classmapping.csv"  
        
        #self.regions = ['slovenia']

        self.load_classmapping(self.classmappingfile)
        
        # Only do the timeseries (breizhcrops) file structure generation once, if a general index doesn't exist
        if not os.path.isfile(self.indexfile):
            self.preprocess()

        self.selected_bands = BANDS

        self.index= pd.read_csv(self.root+os.sep+self.regions[0]+".csv", index_col=None)
        
        for region in self.regions[1:]:
            region_ind = pd.read_csv(self.root+os.sep+region+".csv", index_col=None)
            self.index = pd.concatenate([self.index,region_ind], axis=0)
        
        # self.index will always be all chosen regions summarized
        # index.csv will be the entire index for all existing regions



        self.X_list = None

        self.show_timeseries(0)
        plt.show()

        # if "classid" not in index_region.columns or "classname" not in index_region.columns or "region" not in index_region.columns:
        #     # drop fields that are not in the class mapping
        #     index_region = index_region.loc[index_region["CODE_CULTU"].isin(self.mapping.index)]
        #     index_region[["classid", "classname"]] = index_region["CODE_CULTU"].apply(
        #         lambda code: self.mapping.loc[code])
        #     index_region.to_csv(self.indexfile)

        # # filter zero-length time series
        # if self.index.index.name != "idx":
        #     self.index = self.index.loc[self.index.sequencelength > self.config.filter_length]  # set_index('idx')

        # self.maxseqlength = int(self.index["sequencelength"].max())

    def preprocess(self):

        self.eopatches = [f.name for f in os.scandir(self.root+os.sep+'eopatches') if f.is_dir()]
        self.indexfile = self.root+os.sep+'index.csv'
        print(self.eopatches)
        columns = ['path','eopatch', 'polygon_id','CODE_CULTU', 'sequencelength','classid','classname','region']
        #self.index = pd.DataFrame(columns=columns)
        list_index = list()
        for patch in self.eopatches:
            eop = EOPatch.load(self.root+os.sep+'eopatches'+os.sep+patch)
            polygons = eop.vector_timeless["CROP_TYPE_GDF"]
            for row in polygons.itertuples():
                if row.ct_eu_code not in self.mapping.index.values:
                    continue
                poly_id = int(row.polygon_id)

                classid = self.mapping.loc[row.ct_eu_code].id
                classname = self.mapping.loc[row.ct_eu_code].classname

                list_index.append(
                    {
                        columns[0]:patch+os.sep+str(poly_id), 
                        columns[1]:patch,
                        columns[2]:poly_id,
                        columns[3]:row.ct_eu_code,
                        columns[4]:0,#temp_X.shape[0],
                        columns[5]:classid,
                        columns[6]:classname,
                        columns[7]:''#self.region                  
                    }
                )
                #self.index = pd.concat([self.index, pd.DataFrame([[patch+os.sep+str(poly_id), patch, poly_id, row.ct_eu_code, temp_X.shape[0], classid, classname]], columns=self.index.columns)], axis=0, ignore_index=True)
        self.index = pd.DataFrame(list_index)

        self.split()

        f = {}
        for set in self.split_sets:
            f[set] = h5py.File(self.h5path[set], "w")

        self.index.set_index("path", drop=False, inplace=True)

        for patch in self.eopatches:
            eop = EOPatch.load(self.root+os.sep+'eopatches'+os.sep+patch)
            polygons = eop.vector_timeless["CROP_TYPE_GDF"]
            for row in polygons.itertuples():
                if row.ct_eu_code not in self.mapping.index.values:
                    continue
                poly_id = int(row.polygon_id)
                print(self.index)
                index_row = self.index.loc[patch+os.sep+str(poly_id)]

                polygon = polygons[polygons.polygon_id==poly_id]
                temp = VectorToRasterTask(vector_input=polygon,
                                raster_feature=(FeatureType.MASK_TIMELESS, 'poly'),
                                values=1,
                                raster_shape = (FeatureType.MASK_TIMELESS, 'CROP_TYPE')
                                )
                polygon_indicator_mask = temp.execute(eop).mask_timeless['poly']
                
                # plt.figure(figsize=(10,10))
                # plt.imshow(new_eop.mask_timeless['poly'])
                # plt.show()

                print("num_pixels orig "+str(np.sum(polygon_indicator_mask)))
                seq_length = eop.data["FEATURES_S2"].shape[0]
                num_bands = eop.data["FEATURES_S2"].shape[3]
                
                polygon_indicator_mask_ts = np.repeat(polygon_indicator_mask[np.newaxis,:,:,:], seq_length, axis=0)
                polygon_indicator_mask_ts = np.repeat(polygon_indicator_mask_ts, num_bands, axis=3)

                print(polygon_indicator_mask_ts.shape)
                print("num_pixels "+str(np.sum(polygon_indicator_mask_ts)))
                print("aggregation_test "+str(np.sum(polygon_indicator_mask_ts, axis=(1,2))))
                print("aggregation_test_shape "+str(np.sum(polygon_indicator_mask_ts, axis=(1,2)).shape))

                temp_X=np.sum(np.multiply(polygon_indicator_mask_ts, eop.data["FEATURES_S2"]), axis=(1,2))

                dset = f[index_row.region].create_dataset(patch+os.sep+str(poly_id), data=temp_X)
        self.index.reset_index(inplace=True, drop=True)
        self.write_index()

    def split(self):
        print(self.index)
        X_train, X_test, y_train, y_test  = train_test_split(self.index.values, self.index.classid.values, test_size=0.15, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1) # 0.25 x 0.8 = 0.2

        X_train = pd.DataFrame(X_train, columns=self.index.columns)
        X_train['region'] = 'train'
        X_train.to_csv(self.root+os.sep+'train.csv')
        X_test = pd.DataFrame(X_test, columns=self.index.columns)
        X_test['region'] = 'test'
        X_test.to_csv(self.root+os.sep+'test.csv')
        X_val = pd.DataFrame(X_val, columns=self.index.columns)
        X_val['region'] = 'val'
        X_val.to_csv(self.root+os.sep+'val.csv')

        self.index = pd.concat([X_train, X_val, X_test], ignore_index=True)

        # sort?
        print(self.index)

    def write_index(self):
        self.index.to_csv(self.indexfile)
