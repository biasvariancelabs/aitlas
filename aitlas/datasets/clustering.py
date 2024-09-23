import random
import matplotlib.pyplot as plt
import torch

from ..base import BaseDataset
from ..utils import collate_fn
from .schemas import ClusteringDatasetSchema


"""
Generic dataset for the task of clustering
"""


class ClusteringDataset(BaseDataset):
    """Base clustering dataset class"""

    name = "Clustering Dataset"
    schema = ClusteringDatasetSchema

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def __len__(self):
        return len(self.images)

    def apply_transformations(self, image):
        if self.transform:
            image = self.transform(image)
        return image
    
    def show_image(self, index, show_title=False):
        # plot the image
        img = self[index]
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        return fig

    def show_batch(self, size, show_labels=False):
        if size % 5:
            raise ValueError("The provided size should be divided by 5!")
        image_indices = random.sample(range(0, len(self)), size)
        figure, ax = plt.subplots(
            int(size / 5), 5, figsize=(13.75, 2.8 * int(size / 5))
        )

        for axes, image_index in zip(ax.flatten(), image_indices):
            img = self[image_index]
            axes.imshow(img)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure
