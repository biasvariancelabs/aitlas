import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch
from ..utils import image_invert, image_loader
from .semantic_segmentation import SemanticSegmentationDataset

"""
For the Chactun dataset there is a seperate mask for each label
The object is black and the background is white
"""


class ChactunDataset(SemanticSegmentationDataset):

    labels = ["Aguada", "Building", "Platform"]
    color_mapping = [[255, 255, 0], [100, 100, 100], [0, 255, 0]]
    name = "Chactun"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = np.zeros(
            shape=(image.shape[0], image.shape[1], len(self.masks[index])), dtype=float
        )
        for i, path in enumerate(self.masks[index]):
            mask[:, :, i] = image_invert(path, True) / 255
        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file=None):
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

    def show_image(self, index, show_title=True):
        img, mask = self[index]
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
            img_mask[i][np.where(mask[:, :, i] == 1)] = self.color_mapping[i]

        fig = plt.figure(figsize=(10, 8))
        # if show_title:
        #    fig.suptitle(
        #        f"Image and mask with index {index} from the dataset {self.get_name()}\n",
        #        fontsize=16,
        #        y=1.006,
        #    )
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.3, 1.0, 0.4, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
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
