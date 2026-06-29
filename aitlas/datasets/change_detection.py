import torch
import numpy as np

from ..base import BaseDataset
from .schemas import SegmentationDatasetSchema

"""
Generic dataset for the task of change detection
"""


class ChangeDetectionDataset(BaseDataset):
    """Base class for a change detection dataset"""

    schema = SegmentationDatasetSchema  # Can be reused for now
    labels = None
    name = None

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)

    def __getitem__(self, index):
        """
        A concrete implementation of a change detection dataset should implement this
        and return a tuple of (image1, image2, mask)
        """
        raise NotImplementedError(
            "Your dataset should implement the `__getitem__` method."
        )

    def __len__(self):
        """
        A concrete implementation of a change detection dataset should implement this
        and return the size of the dataset
        """
        raise NotImplementedError(
            "Your dataset should implement the `__len__` method."
        )

    def apply_transformations(self, image1, image2, mask):
        """
        Applies transformations to the images and mask, ensuring synchronization.
        """
        # Apply synchronized geometric transformations
        if self.joint_transform:
            # Stack the two images to pass them as a single "image" to the joint transform.
            # This ensures both images and the mask receive the same geometric augmentations.
            combined_images = np.concatenate([image1, image2], axis=-1)

            # The joint_transform applies augmentations and returns transformed numpy arrays
            transformed_combined, mask = self.joint_transform((combined_images, mask))
        else:
            # If no joint transform, just use the original stacked images
            transformed_combined = np.concatenate([image1, image2], axis=-1)

        # Apply image-only transforms (like ToTensor, Normalization)
        if self.transform:
            # The transform pipeline should handle the stacked image (e.g., 6 channels)
            # and convert it to a tensor.
            transformed_tensor = self.transform(transformed_combined)

            # Split the stacked images back into two separate tensors
            channels = image1.shape[2] if image1.ndim == 3 else 1
            image1, image2 = torch.split(transformed_tensor, channels, dim=0)

        # Apply mask-only transforms
        if self.target_transform:
            mask = self.target_transform(mask)

        # Ensure mask is in the correct format (remove channel dim if it's 1)
        if isinstance(mask, torch.Tensor) and mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        return (image1, image2), mask

    def get_labels(self):
        return self.labels