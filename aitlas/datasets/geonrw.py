import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import imageio
import seaborn as sns

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import GeoNRWSchema


"""
GeoNRW dataset consists of orthorectified aerial photographs, lidar derived DEMs, land cover maps 
with 10 classes and TerraSARX spotlight acquisitions over the German state North Rhine Westphalia.
"""


class GeoNRWDataset(BaseDataset):
    url = "https://ieee-dataport.org/open-access/geonrw"

    labels = ["no data", "forest", "water", "agricultural", "urban", "grassland",
            "railway", "highway", "airport and shipyard", "roads", "buildings"]
    color_mapping = [
        [0, 0, 0],    
        [0, 100, 0],   
        [0, 0, 255],       
        [218, 165, 32],    
        [139, 0, 0],       
        [144, 238, 144],   
        [112, 128, 144],   
        [255, 255, 0],    
        [138, 43, 226],    
        [0, 255, 255],     
        [255, 0, 0]       
    ]

    name = "GeoNRW"
    schema = GeoNRWSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
     
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        self.imagery = self.config.imagery
        self.rgb_images, self.dem_images, self.masks = self.load_dataset(self.data_dir, self.csv_file)


    def __getitem__(self, index):
        if self.imagery == 'aerial':
            image = imageio.imread(self.rgb_images[index]) # 3 or 4 bands image
            if self.selection == 'rgb':
                image = image[:, :, :3]

        elif self.imagery == 'dem':
            image = image_loader(self.dem_images[index])

        elif self.imagery == 'all':
            rgb_image = imageio.imread(self.rgb_images[index]) # 3 or 4 bands image
            if self.selection == 'rgb':
                rgb_image = rgb_image[:, :, :3]

            dem_image = image_loader(self.dem_images[index])

            image = np.dstack((rgb_image, dem_image))

        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v in range(len(self.labels))]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)


    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        rgb_images = []
        dem_images = []
        masks = []

        city_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        if csv_file is not None: # 60/20/20 splits
            df = pd.read_csv(csv_file)
            
            required_columns = {'rgb_path', 'dem_path', 'mask_path'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")
            
            for rgb in df['rgb_path']:
                img_id = rgb[:-8]
                if self.imagery == 'all':
                    if os.path.exists(os.path.join(data_dir, rgb)) and os.path.exists(os.path.join(data_dir, f'{img_id}_seg.tif')) and os.path.exists(os.path.join(data_dir, f'{img_id}_dem.tif')):
                        rgb_images.append(os.path.join(data_dir, rgb))
                        dem_images.append(os.path.join(data_dir, f'{img_id}_dem.tif'))
                        masks.append(os.path.join(data_dir, f'{img_id}_seg.tif'))
                elif self.imagery == 'aerial':
                    if os.path.exists(os.path.join(data_dir, rgb)) and os.path.exists(os.path.join(data_dir, f'{img_id}_seg.tif')):
                        rgb_images.append(os.path.join(data_dir, rgb))
                        masks.append(os.path.join(data_dir, f'{img_id}_seg.tif'))
                elif self.imagery == 'dem':
                    if os.path.exists(os.path.join(data_dir, f'{img_id}_dem.tif')) and os.path.exists(os.path.join(data_dir, f'{img_id}_seg.tif')):
                        dem_images.append(os.path.join(data_dir, f'{img_id}_dem.tif'))
                        masks.append(os.path.join(data_dir, f'{img_id}_seg.tif'))
        else:
            for city in city_dirs:
                rgb_files = glob.glob(os.path.join(city, "*_rgb.jp2"))
                for rgb_path in rgb_files:
                    base_id = os.path.basename(rgb_path).replace("_rgb.jp2", "")
                    dem_path = os.path.join(city, f"{base_id}_dem.tif")
                    seg_path = os.path.join(city, f"{base_id}_seg.tif")

                    if not (os.path.exists(dem_path) and os.path.exists(seg_path)):
                        continue

                    rgb_images.append(rgb_path)
                    dem_images.append(dem_path)
                    masks.append(seg_path)

        return rgb_images, dem_images, masks

              
    def __len__(self):
        return len(self.masks)

    def get_labels(self):
        return self.labels
    
    def apply_transformations(self, image, mask):
        if self.joint_transform:
            image, mask = self.joint_transform((image, mask))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}

        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[label] += mask[:, :, :, index].sum().item()

        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
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
    

    def show_image(self, index, show_title=False):
        image, mask = self[index]
        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=label,
                )
            )
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(16, 6))

        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, 1.0, 0.6, 0.2),
                ncol=3, mode='expand', loc='lower left', prop={'size': 12})

        if self.imagery == 'aerial' or self.imagery == 'dem':
            # Image
            plt.subplot(1, 3, 1)
            plt.imshow(image) 
            plt.title("Image")
            plt.axis("off")

            # Segmentation Mask
            plt.subplot(1, 3, 3)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")

        elif self.imagery == 'all':
            # RGB Image
            plt.subplot(1, 3, 1)
            plt.imshow(image[:,:,:-1])
            plt.title("RGB Image")
            plt.axis("off")

            # DEM Image
            plt.subplot(1, 3, 2)
            plt.imshow(image[:,:,-1]) 
            plt.title("DEM")
            plt.axis("off")

            # Segmentation Mask
            plt.subplot(1, 3, 3)
            plt.imshow(img_mask)
            plt.title("Mask")
            plt.axis("off")

        fig.tight_layout()
        plt.show()
        return fig