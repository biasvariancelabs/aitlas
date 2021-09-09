import os
import numpy as np
import matplotlib.pyplot as plt

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema
from matplotlib.patches import Patch

#"Background": 0
#"Buildings": 1

LABELS = ["Background", "Forest"]
# Color mapping for the labels
COLOR_MAPPING = [[0, 0, 0], [255, 255, 255]]

"""
The Amazon Rainforest dataset for semantic segmentation 
contains GeoTIFF images with 512x512 pixels and associated PNG masks 
(forest indicated in white and non-forest in black color)
"""


class AmazonRainforestDataset(BaseDataset):
    url = "https://zenodo.org/record/3233081#.YTYm_44zaUk"

    schema = SegmentationDatasetSchema
    labels = LABELS
    color_mapping = COLOR_MAPPING
    name = "Amazon Rainforest dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema
        BaseDataset.__init__(self, config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.root)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], False)
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype('float32')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, root_dir):
        if not self.labels:
            raise ValueError(
                "You need to provide the list of labels for the dataset"
            )

        ids = os.listdir(os.path.join(root_dir, 'images'))
        self.images = [os.path.join(root_dir, 'images', image_id) for image_id in ids]
        self.masks = [os.path.join(root_dir, 'masks', image_id) for image_id in ids]

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

