import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.patches import Patch
from skimage.transform import resize

from ..base import BaseDataset
from .schemas import MSBCSchema
from ..utils import image_loader


"""
The multisource OSCD (MSOSCD) dataset contains multispectral, SAR, and VHR multisource data for change detection. 
It is reformed from OSCD dataset.
"""


def resize_to_match(src, ref, order=1):
    """
    Resize source image to match spatial size of reference.
    src, ref: (H, W, C)
    """
    return resize(
        src,
        (ref.shape[0], ref.shape[1], src.shape[2]),
        order=order,           
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    ).astype(src.dtype)


class MSOSCDDataset(BaseDataset):

    url = "https://github.com/Lihy256/MSCDUnet"
    name = "MSOSCD"
    schema = MSBCSchema
    labels = ["no change","change"]
    color_mapping =  [[0,0,0],[255, 255, 255]]


    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.mode = self.config.mode
        self.selection = self.config.selection
        self.imagery = self.config.imagery
        self.bands_s1 = self.config.bands_s1
        self.bands_s2 = self.config.bands_s2
        self.data = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        if self.csv_file == None: # original splits
            img_id = self.data[index]
            mode = self.mode # train, validation or test 

        elif self.csv_file: # new 60/20/20 splits
            img_id = self.data[index][0]
            mode = self.data[index][1] # original split

        rgb_1 = image_loader(os.path.join(self.data_dir, mode, 'rgb1', f'{img_id}.tif'))
        rgb_2 = image_loader(os.path.join(self.data_dir, mode, 'rgb2', f'{img_id}.tif'))
        # normalize images
        rgb_1 = (rgb_1 - rgb_1.min()) / (rgb_1.max() - rgb_1.min() + 1e-8)
        rgb_2 = (rgb_2 - rgb_2.min()) / (rgb_2.max() - rgb_2.min() + 1e-8)

        if self.imagery == "s1" or self.imagery == "all":
            # SAR has 4 bands. The first 2 bands are VV, VH of time 1, and the later 2 bands are VV, VH of time 2.
            # time 1
            sar_vv_1 = image_loader(os.path.join(self.data_dir, mode, 'sar', f'{img_id}.tif'))[:,:,0]
            sar_vh_1 = image_loader(os.path.join(self.data_dir, mode, 'sar', f'{img_id}.tif'))[:,:,1]
            sar_vv_vh_1 = sar_vv_1/sar_vh_1

            # time 2
            sar_vv_2 = image_loader(os.path.join(self.data_dir, mode, 'sar', f'{img_id}.tif'))[:,:,2]
            sar_vh_2 = image_loader(os.path.join(self.data_dir, mode, 'sar', f'{img_id}.tif'))[:,:,3]
            sar_vv_vh_2 = sar_vv_2/sar_vh_2

            if self.selection == "rgb":
                sar_1 = np.dstack((sar_vv_1, sar_vh_1, sar_vv_vh_1))
                sar_2 = np.dstack((sar_vv_2, sar_vh_2, sar_vv_vh_2))
                # normalize image
                sar_1 = (sar_1 - sar_1.min()) / (sar_1.max() - sar_1.min() + 1e-8)
                sar_2 = (sar_2 - sar_2.min()) / (sar_2.max() - sar_2.min() + 1e-8)
            elif self.selection == "all":
                sar_1 = np.dstack((sar_vv_1, sar_vh_1))
                sar_2 = np.dstack((sar_vv_2, sar_vh_2))
                # normalize image
                sar_1 = (sar_1 - sar_1.min()) / (sar_1.max() - sar_1.min() + 1e-8)
                sar_2 = (sar_2 - sar_2.min()) / (sar_2.max() - sar_2.min() + 1e-8)
            elif self.selection == "bands":
                if self.bands_s1 == None:
                    print("The config must contain a bands_s1 field with a list of chosen bands")
                sar_1 = np.dstack((sar_vv_1, sar_vh_1))
                sar_2 = np.dstack((sar_vv_2, sar_vh_2))
                idx = []
                s1_bands = ["VV","VH"]
                for b in self.bands_s1:
                    if b not in s1_bands:
                        print("The bands must be valid and available Sentinel-1 bands, i.e. one of VV, VH")
                        break
                    else:
                        idx.append(s1_bands.index(b))
                sar_1 = sar_1[:,:,idx]
                sar_2 = sar_2[:,:,idx]
                # normalize image
                sar_1 = (sar_1 - sar_1.min()) / (sar_1.max() - sar_1.min() + 1e-8)
                sar_2 = (sar_2 - sar_2.min()) / (sar_2.max() - sar_2.min() + 1e-8)

        if self.imagery == "s2" or self.imagery == "all":
            # OPT has 14 bands. The first 7 bands are B2,B3,B4,B8,B8A,B11,B12 of time 1, and the later 7 bands are B2,B3,B4,B8,B8A,B11,B12 of time 2.
            if self.selection == 'rgb':
                opt_1 = image_loader(os.path.join(self.data_dir, mode, 'opt', f'{img_id}.tif'))[:,:,:3]
                opt_2 = image_loader(os.path.join(self.data_dir, mode, 'opt', f'{img_id}.tif'))[:,:,7:10]
                # normalize images
                opt_1 = (opt_1 - opt_1.min()) / (opt_1.max() - opt_1.min() + 1e-8)
                opt_2 = (opt_2 - opt_2.min()) / (opt_2.max() - opt_2.min() + 1e-8)
            elif self.selection == "all":
                opt_1 = image_loader(os.path.join(self.data_dir, mode, 'opt', f'{img_id}.tif'))[:,:,:7]
                opt_2 = image_loader(os.path.join(self.data_dir, mode, 'opt', f'{img_id}.tif'))[:,:,7:14]
                # normalize images
                opt_1 = (opt_1 - opt_1.min()) / (opt_1.max() - opt_1.min() + 1e-8)
                opt_2 = (opt_2 - opt_2.min()) / (opt_2.max() - opt_2.min() + 1e-8)
            elif self.selection == "bands": 
                if self.bands_s2 == None:
                    print("The config must contain a bands_s2 field with a list of chosen bands")
                opt_1 = image_loader(os.path.join(self.data_dir, mode, 'opt', f'{img_id}.tif'))[:,:,:7]
                opt_2 = image_loader(os.path.join(self.data_dir, mode, 'opt', f'{img_id}.tif'))[:,:,7:14]
                idx = []
                s2_bands = ["B02","B03","B04","B08","B8A","B11","B12"]
                for b in self.bands_s2:
                    if b not in s2_bands:
                        print("The bands must be valid Sentinel-2 bands, i.e. one of B02, B03, B04, B08, B8A, B11, B12")
                        break
                    else:
                        idx.append(s2_bands.index(b))
                opt_1 = opt_1[:,:,idx]
                opt_2 = opt_2[:,:,idx]
                # normalize images
                opt_1 = (opt_1 - opt_1.min()) / (opt_1.max() - opt_1.min() + 1e-8)
                opt_2 = (opt_2 - opt_2.min()) / (opt_2.max() - opt_2.min() + 1e-8)


        mask = image_loader(os.path.join(self.data_dir, mode, 'label', f'{img_id}.tif'))
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        if self.imagery == "all": 
            # the images have a different resolution (dstack won't work)
            # solution: resample the rgb and s1 images to the resolution of s2
            rgb_1 = resize_to_match(rgb_1, opt_1)
            rgb_2 = resize_to_match(rgb_2, opt_2)

            sar_1 = resize_to_match(sar_1, opt_1)
            sar_2 = resize_to_match(sar_2, opt_2)

            image_1, image_2 = np.dstack((sar_1, opt_1, rgb_1)), np.dstack((sar_2, opt_2, rgb_2))
          
        elif self.imagery == "aerial":
            image_1, image_2 = rgb_1, rgb_2
        elif self.imagery == "s1":
            image_1, image_2 = sar_1, sar_2
        elif self.imagery == "s2":
            image_1, image_2 = opt_1, opt_2

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return (image_1, image_2), mask


    def load_dataset(self, data_dir):
        if self.csv_file == None: # original train/val/test splits
            self.data = []
            for id in os.listdir(os.path.join(data_dir, self.mode, 'rgb1')):
                self.data.append(id[: -4])

        elif self.csv_file: # new 60/20/20 train/val/test splits provided by BVLabs
            self.data = []
            with open(self.csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None) # skip header
                for row in reader:
                    img_id = row[0] 
                    original_split = row[1]
                    self.data.append((img_id, original_split))

        return self.data

    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.labels

    def show_image(self, index, show_title=False):
        if self.selection != "rgb":
            print("The selection paramater should be set to rgb for image visualization")
        if self.imagery == "all":
            data = self[index]
            (image_1, image_2), mask = data

            sar_1 = image_1[:,:,:3]
            opt_1 = image_1[:,:,3:6]
            rgb_1 = image_1[:,:,6:]

            sar_2 = image_2[:,:,:3]
            opt_2 = image_2[:,:,3:6]
            rgb_2 = image_2[:,:,6:]

            # Create mask visualization
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

            fig = plt.figure(figsize=(15, 6))
            axes = fig.subplots(1, 7)

            # Add legend at the top
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.85, 0.8, 0.05), ncol=3,
                        loc='center', prop={'size': 12})

            axes[0].imshow(img_mask)
            axes[0].set_title("Mask")
            axes[0].axis("off")

            axes[1].imshow(sar_1)
            axes[1].set_title("S1 image t1")
            axes[1].axis("off")

            axes[2].imshow(sar_2)
            axes[2].set_title("S1 image t2")
            axes[2].axis("off")

            axes[3].imshow(opt_1)
            axes[3].set_title("S2 image t1")
            axes[3].axis("off")

            axes[4].imshow(opt_2)
            axes[4].set_title("S2 image t2")
            axes[4].axis("off")

            axes[5].imshow(rgb_1)
            axes[5].set_title("RGB image t1")
            axes[5].axis("off")

            axes[6].imshow(rgb_2)
            axes[6].set_title("RGB image t2")
            axes[6].axis("off")

        elif self.imagery != "all":
            data = self[index]
            (image_1, image_2), mask = data

            # Create mask visualization
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

            fig = plt.figure(figsize=(15, 6))
            axes = fig.subplots(1, 3)

            # Add legend at the top
            fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.0, 0.8, 0.05), ncol=3,
                        loc='center', prop={'size': 12})

            axes[0].imshow(img_mask)
            axes[0].set_title("Mask")
            axes[0].axis("off")

            axes[1].imshow(image_1)
            axes[1].set_title("Image t1")
            axes[1].axis("off")

            axes[2].imshow(image_2)
            axes[2].set_title("Image t2")
            axes[2].axis("off")

        fig.tight_layout()
        plt.show()
        return fig


    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.data)):
            image, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig