import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema

#"Background": 0
#"Buildings": 1

LABELS = ["Background", "Buildings"]
# Color mapping for the labels
COLOR_MAPPING = [[0, 0, 0], [255, 255, 255]]

"""
The training set contains 180 color image tiles of size 5000×5000, covering a surface of 1500 m × 1500 m each 
(at a 30 cm resolution). The format is GeoTIFF (TIFF with georeferencing, but the images can be used as any other TIFF). 
The reference data is in a different folder and the file names correspond exactly to those of the color images. 
In the case of the reference data, the tiles are single-channel images 
with values 255 for the building class and 0 for the not building class.
Use function split_images from utils to split the images and the masks in smaller patches
"""


class InriaDataset(BaseDataset):
    url = "https://project.inria.fr/aerialimagelabeling/"

    schema = SegmentationDatasetSchema
    labels = LABELS
    color_mapping = COLOR_MAPPING
    name = "Inria"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.root, self.config.csv_file_path)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], True) / 255
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype('float32')
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, root_dir, file_path):
        if not self.labels:
            raise ValueError(
                "You need to provide the list of labels for the dataset"
            )
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for index, row in enumerate(csv_reader):
                self.images.append(os.path.join(root_dir, row[0] + '.jpg'))
                self.masks.append(os.path.join(root_dir, row[0] + '_m.png'))

    def get_labels(self):
        return self.labels

    def show_image(self, index):
        img = self[index][0]
        mask = self[index][1]
        img_mask = np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8)
        legend_elements = []
        for i, label in enumerate(self.labels):
            legend_elements.append(Patch(facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                                         label=self.labels[i]))
            img_mask[np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        fig.legend(handles=legend_elements)
        plt.title(f"Image and mask with index {index} from the dataset {self.get_name()}\n", fontsize=14)
        plt.subplot(1, 2, 1)
        plt.imshow(img_mask)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return fig
