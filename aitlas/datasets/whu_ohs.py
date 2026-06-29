import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from matplotlib.patches import Patch
import math

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import WHUOHSSchema

"""
The dataset consists of about 90 million manually labeled samples of 7795 Orbita hyperspectral satellite (OHS) 
image patches from 40 Chinese locations.
"""


class WHUOHSDataset(SemanticSegmentationDataset):
    url = "http://irsip.whu.edu.cn/resources/WHU_OHS_show.php"

    labels = ["no data","paddy field","dry farm","woodland","shrubbery","sparse woodland","other forest land",
              "high-covered grrassland","medium-covered grassland","low-covered grassland","river canal","lake",
              "reservoir pond","beach land","shoal","urban built-up","rural settlement","other constructionh land",
              "sand","gobi","saline-alkali soil","marshland","bare land","bare rock","ocean"]
    color_mapping = [[0,0,0],[193,208,255],[0,255,195],[39,115,0],[160,255,113],[74,230,2],[84,255,1],[118,116,4],
                     [167,169,0],[252,255,1],[116,178,251],[1,91,233],[0,39,115],[123,142,245],[0,167,134],[110,1,0],
                     [254,127,129],[253,193,191],[253,191,232],[254,1,195],[228,2,169],[166,1,127],[110,2,77],[253,116,219],
                     [162,160,159]] 
    name = "WHU-OHS"
    schema = WHUOHSSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.bands_ohs = self.config.bands_ohs

    def __getitem__(self, index):
        image = image_loader(self.images[index])

        # to consider 3-band RGB images
        if self.selection == 'rgb':
            Red = image[:,:,13]
            Green = image[:,:,6]
            Blue = image[:,:,1]
            image = np.dstack([Red,Green,Blue])/10000 #pixel values were scaled by 10000

        # to consider all the 32 bands
        elif self.selection == 'all':
            image = image/10000

        elif self.selection == 'bands':
            image = image/10000
            if self.bands_ohs == None:
                print("The config must contain a bands_ohs field with a list of chosen bands")
            idx = []
            ohs_bands = ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10","B11","B12","B13","B14","B15","B16","B17","B18","B19","B20","B21","B22","B23","B24","B25","B26","B27","B28","B29","B30","B31","B32"]
            for b in self.bands_ohs:
                if b not in ohs_bands:
                    print("The bands must be valid OHS bands, i.e. one of B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31, B32")
                    break
                else:
                    idx.append(ohs_bands.index(b))
            image = image[:,:,idx]

        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        # the original dataset is split into train/val/test set of 4648/513/2459
        # csv_files created by BVLabs provide a 60%/20%/20% split of 4572/1524/1524
        if csv_file:
            ids = []
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                for row in reader:
                    file_name = row[0].split('/')[1]
                    ids.append(file_name)
                    
        else: # original split
            ids = os.listdir(os.path.join(data_dir, "images"))

        self.images = [os.path.join(data_dir, "images", image_id) for image_id in ids]
        self.masks = [os.path.join(data_dir, "labels", image_id) for image_id in ids]

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
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
    
    def show_image(self, index, show_title=False):
        img, mask = self[index]
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

        fig = plt.figure(figsize=(10, 8))
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.1, height_factor, 0.8, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig