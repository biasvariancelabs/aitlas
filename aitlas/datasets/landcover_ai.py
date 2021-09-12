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
#"Woodlands": 2
#"Water": 3
#"Road": 4

LABELS = ["Background", "Buildings", "Woodlands", "Water", "Road"]
# Color mapping for the labels
COLOR_MAPPING = [[255, 255, 0], [0, 0, 0], [0, 255, 0], [0, 0, 255], [200, 200, 200]]

"""
41 orthophoto tiles from different counties located in all regions. Every tile has about 5 km2. 
There are 33 images with resolution 25cm (ca. 9000 × 9500 px) and 8 images with resolution 50cm (ca. 4200 × 4700 px)
Tne masks are codded with building (1), woodland (2), water (3), and road (4)
"""


class LandCoverAiDataset(BaseDataset):
    url = "https://landcover.ai/"

    schema = SegmentationDatasetSchema
    labels = LABELS
    color_mapping = COLOR_MAPPING
    name = "Landcover AI dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.root, self.config.csv_file_path)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], True)
        # extract certain classes from mask (e.g. Buildings)
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype('float32')
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
