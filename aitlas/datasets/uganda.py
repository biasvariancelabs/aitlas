import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..base import BaseDataset
from ..utils import image_invert, image_loader
from .schemas import SegmentationDatasetSchema


LABELS = ["Buildings", "Both", "Roads"]
COLOR_MAPPING = [[255, 255, 0], [100, 100, 100], [0, 255, 0]]


"""
For the Chactun dataset there is a seperate mask for each label
The object is black and the background is white
"""


class UgandaDataset(BaseDataset):
    """Load only first 3 bands (RGB) of the sample image."""

    schema = SegmentationDatasetSchema
    labels = LABELS
    color_mapping = COLOR_MAPPING
    name = "Uganda"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.data_dir)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = np.zeros(
            shape=(len(self.masks[index]), image.shape[0], image.shape[1]), dtype=float
        )
        for i, path in enumerate(self.masks[index]):
            mask[i] = image_invert(path, False)  # if true, inverts an 8 bit image: (255 - i), conversion to [0, 1] is done in Transfomrs
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, data_dir):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        masks_for_image = []
        for root, _, fnames in sorted(os.walk(data_dir)):
            for i, fname in enumerate(sorted(fnames)):
                path = os.path.join(data_dir, fname)
                if i % 4 == 0:
                    self.images.append(path)
                    masks_for_image = []
                else:
                    masks_for_image.append(path)
                    if i % 4 == 3:
                        self.masks.append(masks_for_image)

    def get_labels(self):
        return self.labels

    def show_image(self, index):
        img = self[index][0]
        mask = self[index][1].transpose(1, 2, 0)
        legend_elements = []
        img_mask = []
        for i, label in enumerate(self.labels):
            legend_elements.append(
                Patch(
                    facecolor=tuple([x / 255 for x in self.color_mapping[i]]),
                    label=self.labels[i],
                )
            )
            img_mask.append(np.zeros([mask.shape[0], mask.shape[1], 3], np.uint8))
            img_mask[i][np.where(mask[:, :, i] == 255)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(
            f"Image and mask with index {index} from the dataset {self.get_name()}\n",
            fontsize=16,
            y=1.006,
        )
        fig.legend(
            handles=legend_elements, bbox_to_anchor=[0.95, 0.95], loc="upper right"
        )
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(2, 2, 2)
        plt.imshow(img_mask[0])
        plt.axis("off")
        plt.subplot(2, 2, 3)
        plt.imshow(img_mask[1])
        plt.axis("off")
        plt.subplot(2, 2, 4)
        plt.imshow(img_mask[2])
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig

