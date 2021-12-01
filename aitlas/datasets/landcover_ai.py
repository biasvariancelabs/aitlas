import csv
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema


LABELS = ["Background", "Buildings", "Woodlands", "Water", "Road"]
# Color mapping for the labels
COLOR_MAPPING = [[255, 255, 0], [0, 0, 0], [0, 255, 0], [0, 0, 255], [200, 200, 200]]

"""
41 orthophoto tiles from different counties located in all regions. Every tile has about 5 km2.
There are 33 images with resolution 25cm (ca. 9000 × 9500 px) and 8 images with resolution 50cm (ca. 4200 × 4700 px)
Tne masks are codded with building (1), woodland (2), water (3), and road (4)
Use function split_images to split the images and the masks in smaller patches
"""


class LandCoverAiDataset(BaseDataset):
    url = "https://landcover.ai/"

    schema = SegmentationDatasetSchema
    labels = LABELS
    color_mapping = COLOR_MAPPING
    name = "Landcover AI"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.data_dir, self.config.csv_file)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], True)
        # extract certain classes from mask (e.g. Buildings)
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        with open(csv_file, "r") as f:
            csv_reader = csv.reader(f)
            for index, row in enumerate(csv_reader):
                self.images.append(os.path.join(data_dir, row[0] + ".jpg"))
                self.masks.append(os.path.join(data_dir, row[0] + "_m.png"))

    def get_labels(self):
        return self.labels

    def show_image(self, index):
        img = self[index][0]
        mask = self[index][1]
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
        fig.suptitle(
            f"Image and mask with index {index} from the dataset {self.get_name()}\n",
            fontsize=16,
            y=1.006,
        )
        fig.legend(handles=legend_elements, bbox_to_anchor=[0.5, 0.85], loc="center")
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig


def split_images(imgs_dir, masks_dir, output_dir):
    target_size = 512

    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))

    img_paths.sort()
    mask_paths.sort()

    os.makedirs(output_dir)
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], target_size):
            for x in range(0, img.shape[1], target_size):
                img_tile = img[y : y + target_size, x : x + target_size]
                mask_tile = mask[y : y + target_size, x : x + target_size]

                if (
                    img_tile.shape[0] == target_size
                    and img_tile.shape[1] == target_size
                ):
                    out_img_path = os.path.join(
                        output_dir, "{}_{}.jpg".format(img_filename, k)
                    )
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(
                        output_dir, "{}_{}_m.png".format(mask_filename, k)
                    )
                    cv2.imwrite(out_mask_path, mask_tile)

                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
