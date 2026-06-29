import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import seaborn as sns
import csv
from skimage.transform import resize

from ..utils import image_loader
from .schemas import PASTISHDSchema
from .semantic_segmentation import SemanticSegmentationDataset


'''
PASTIS-HD is a high-resolution multimodal benchmark dataset for semantic segmentation of agricultural parcels from 
satellite image time series. It combines 10m-resolution Sentinel-2 optical bands, SAR (VV and VH) from Sentinel-1, 
and very high-resolution (VHR) SPOT 6-7 RGB imagery. Each instance includes a time series with multi-sensor signals 
and corresponding semantic labels.
'''

class PASTISHDDataset(SemanticSegmentationDataset):
    schema = PASTISHDSchema
    name = "PASTIS HD"
    url = "https://huggingface.co/datasets/IGNF/PASTIS-HD"
    labels = [
        "Background",
        "Meadow",
        "Soft winter wheat",
        "Corn",
        "Winter barley",
        "Winter rapeseed",
        "Spring barley",
        "Sunflower",
        "Grapevine",
        "Beet",
        "Winter triticale",
        "Winter durum wheat",
        "Fruits vegetables flowers",
        "Potatoes",
        "Leguminous fodder",
        "Soybeans",
        "Orchard",
        "Mixed cereal",
        "Sorghum",
        "Void label"
    ]
    color_mapping = [
        [0, 0, 0],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [255, 255, 255]
    ]

    def __init__(self, config):
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.bands_s2 = self.config.bands_s2
        self.bands_s1 = self.config.bands_s1

    def __getitem__(self, index):
        # the data contains only one SPOT image per patch but time series for S2 and S1
        id = self.data[index]
        
        # SPOT image
        spot_image = image_loader(os.path.join(self.data_dir, "DATA_SPOT", "PASTIS_SPOT6_RVB_1M00_2019",f"SPOT6_RVB_1M00_2019_{id}.tif")) # shape = (height, width, band)

        # Sentinel-2
        if self.imagery == "S2" or self.imagery == "all":
            s2_image = np.load(os.path.join(self.data_dir, "DATA_S2", f"S2_{id}.npy")) # (timestamp, band, height, width)
            image_list_s2 = []
            for t in range(s2_image.shape[0]): # S2 has 10 bands
                if self.selection == "rgb":
                    time_step_image = np.transpose(s2_image[t, [2,1,0]], (1, 2, 0))
                elif self.selection == "all":
                    time_step_image = np.transpose(s2_image[t, :], (1, 2, 0))
                elif self.selection == "bands": 
                    if self.bands_s2 == None:
                        print("The config must contain a bands_s2 field with a list of chosen bands")
                    idx = []
                    s2_bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
                    for b in self.bands_s2:
                        if b not in s2_bands:
                            print("The bands must be valid Sentinel-2 bands, i.e. one of B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12")
                            break
                        else:
                            idx.append(s2_bands.index(b))
                    time_step_image = np.transpose(s2_image[t, idx], (1, 2, 0))

                image_list_s2.append(time_step_image)
            s2 = np.array(image_list_s2) # (time, height, width, band)

        # Sentinel-1 ascending
        if self.imagery == "S1A" or self.imagery == "all":
            s1a_image = np.load(os.path.join(self.data_dir, "DATA_S1A", f"S1A_{id}.npy"),allow_pickle=True) # (timestamp, band, height, width)
            image_list_s1a = []
            for t in range(s1a_image.shape[0]): # S1 has VV, VH and VV/VH bands
                if self.selection == "rgb" or self.selection == "all":
                    time_step_image_s1a = np.transpose(s1a_image[t, 0:3], (1, 2, 0))
                elif self.selection == "bands": 
                    if self.bands_s1 == None:
                        print("The config must contain a bands_s1 field with a list of chosen bands")
                    idx = []
                    s1_bands = ["VV","VH","VV/VH"]
                    for b in self.bands_s1:
                        if b not in s1_bands:
                            print("The bands must be valid and available Sentinel-1 bands, i.e. one of VV, VH, VV/VH")
                            break
                        else:
                            idx.append(s1_bands.index(b))
                    time_step_image_s1a = np.transpose(s1a_image[t, idx], (1, 2, 0)) 
                image_list_s1a.append(time_step_image_s1a)
            s1a = np.array(image_list_s1a) # (time, height, width, band)

        # Sentinel-1 descending
        if self.imagery == "S1D" or self.imagery == "all":
            s1d_image = np.load(os.path.join(self.data_dir, "DATA_S1D", f"S1D_{id}.npy")) # (timestamp, band, height, width)
            image_list_s1d = []
            for t in range(s1d_image.shape[0]): # S1 has VV, VH and VV/VH bands
                if self.selection == "rgb" or self.selection == "all":
                    time_step_image_s1d = np.transpose(s1d_image[t, 0:3], (1, 2, 0))
                elif self.selection == "bands": 
                    if self.bands_s1 == None:
                        print("The config must contain a bands_s1 field with a list of chosen bands")
                    idx = []
                    s1_bands = ["VV","VH","VV/VH"]
                    for b in self.bands_s1:
                        if b not in s1_bands:
                            print("The bands must be valid and available Sentinel-1 bands, i.e. one of VV, VH, VV/VH")
                            break
                        else:
                            idx.append(s1_bands.index(b))
                    time_step_image_s1d = np.transpose(s1d_image[t, idx], (1, 2, 0)) 
                image_list_s1d.append(time_step_image_s1d)
            s1d = np.array(image_list_s1d) # (time, height, width, band)
    
        if self.imagery == 'all': # considers only the first timestamp for each patch
            # resize the spot image so the shapes match for concatenation
            reference = np.load(os.path.join(self.data_dir, "DATA_S2", f"S2_{id}.npy"))[0,0,:,:] # S2 B02 of timestep 1
            target_shape = reference.shape
            if spot_image.shape != target_shape:
                spot_image = resize(spot_image, target_shape, order=1, preserve_range=True).astype(spot_image.dtype)

            # normalize images
            s1a = s1a[0,:,:,:] # first time step
            s1a = (s1a - s1a.min()) / (s1a.max() - s1a.min() + 1e-8)
            s2 = s2[0,:,:,:] # first time step
            s2 = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-8)
            spot_image = (spot_image - spot_image.min()) / (spot_image.max() - spot_image.min() + 1e-8)

            image = np.dstack((s1a, s2, spot_image))

        # for simplicity we keep only the firt time stamp
        elif self.imagery == 'S2':
            image = s2[0,:,:,:] # can be changed to image = s2 to consider the whole S2 time series
        elif self.imagery == 'S1A':
            image = s1a[0,:,:,:] # can be changed to image = s1a to consider the whole S1A time series
        elif self.imagery == 'S1D':
            image = s1d[0,:,:,:] # can be changed to image = s1d to consider the whole S1D time series
        elif self.imagery == 'SPOT':
            image = spot_image[0,:,:,:]
            
        if self.transform:
            image = self.transform(image)

        # mask image
        mask = np.load(os.path.join(self.data_dir, "ANNOTATIONS", f"TARGET_{id}.npy"))

        if len(mask.shape) == 3:
            semantic_mask = mask[0]
        else:
            semantic_mask = mask

        num_classes = len(self.labels)
        mask = np.eye(num_classes)[semantic_mask]
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def load_dataset(self, data_dir, csv_file=None):
        # Find IDs which have both S2 data and annotations (not all images are annotated)
        s2_dir = os.path.join(data_dir, "DATA_S2")
        annotation_dir = os.path.join(data_dir, "ANNOTATIONS")

        s2_files = os.listdir(s2_dir)
        s2_ids = [os.path.splitext(f)[0] for f in s2_files if f.endswith('.npy')]
        s2_ids = [id_name.split('_')[-1] for id_name in s2_ids if '_' in id_name]

        annotation_files = os.listdir(annotation_dir)
        annotation_ids = [os.path.splitext(f)[0] for f in annotation_files if f.endswith('.npy')]
        annotation_ids = [id_name.split('_')[-1] for id_name in annotation_ids if '_' in id_name]

        common_ids = list(set(s2_ids).intersection(set(annotation_ids)))
        self.data = common_ids

        # csv_files with a 60/20/20 train/val/test split are provided by BVLabs
        # the csv_files contain the annotation file names
        if csv_file:
            self.data = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    file_name = row[0][7 : -4]  
                    self.data.append(file_name)

    def __len__(self):
        return len(self.data)

    def show_image(self, index, show_title=False):
        # Shows the first image of the time series
        if self.selection != "rgb":
            print("The selection parameter must be set to rgb for visualization")
       
        image, mask = self[index]

        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        if self.imagery == "SPOT":

            fig = plt.figure(figsize=(10, 8))
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.8, 0.8, 0.2), ncol=3, mode='expand',
                    loc='lower left', prop={'size': 12})
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")
            fig.tight_layout()
            plt.show()

        elif self.imagery == "S2" or self.imagery == "S1A" or self.imagery == "S1D":
            image = image[0,:,:,:] # first time step
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = image.astype(np.float64)

            fig = plt.figure(figsize=(10, 8))
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.8, 0.8, 0.2), ncol=3, mode='expand',
                    loc='lower left', prop={'size': 12})
            plt.subplot(1, 2, 1)
            plt.imshow(image) 
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_mask)
            plt.axis("off")
            fig.tight_layout()
            plt.show()

        elif self.imagery == "all":
            fig, axes = plt.subplots(1, 4, figsize=(16, 8))
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, 0.8, 0.6, 0.2), ncol=3, mode='expand',
                loc='lower left', prop={'size': 12})

            axes[0].imshow(image[:,:,:3])
            axes[0].set_title('Sentinel-1', fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(image[:,:,3:6])
            axes[1].set_title('Sentinel-2', fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(image[:,:,6:])
            axes[2].set_title('SPOT', fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(img_mask)
            axes[3].set_title('Mask', fontsize=12)
            axes[3].axis("off")

            fig.tight_layout()
            plt.show()

        return fig

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in range(len(self)):
            _, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()

        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count
    
    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        fig.autofmt_xdate()
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig
    
    def show_timeseries(self, index, longitude, latitude):
        if self.selection != "all":
            print("The selection parameter should be set to 'all' for time series visualization.")
  
        if self.imagery == "SPOT" or self.imagery == "all":
            print("The imagery parameter should be set to S1A, S1D or S2. SPOT imagery does not contain multitemporal data.")
        
        timeseries, mask = self[index]

        crop = mask[longitude, latitude]
        crop_name = "Unknown"
        for j, k in enumerate(self.labels):
            if crop[j] == 1:
                crop_name = k
                break

        time_range = range(len(timeseries))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Time series with index {index} from the region with pixel coordinates ({longitude},{latitude}), with label {crop_name}\n")

        if self.imagery == 'S1A' or self.imagery == 'S1D':
            # extract S1A bands for a specific pixel
            s1_pixel = []
            for b in range(3):
                s1_band_series = [timeseries[i][longitude, latitude, b] for i in time_range]
                s1_pixel.append(s1_band_series)

            for i, s1_band_series in enumerate(s1_pixel):
                if i == 0:
                    band = 'VV'
                elif i == 1:
                    band = 'VH'
                else: 
                    band = 'VV/VH'
                ax.plot(time_range, s1_band_series, label=f"S1A {band}")

        elif self.imagery == 'S2':
            # extract Sentinel-2 bands
            s2_pixel = []
            for b in range(10):
                s2_band_series = [timeseries[i][longitude, latitude, b] for i in time_range]
                s2_pixel.append(s2_band_series)

            for i, s2_band_series in enumerate(s2_pixel):
                ax.plot(time_range, s2_band_series, label=f"S2-B{i+1}", alpha=0.5)

        ax.set_xlabel("Time step")
        ax.set_ylabel("Pixel value")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig