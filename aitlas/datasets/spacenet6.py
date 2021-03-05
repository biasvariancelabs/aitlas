"""
Notes
-----
    Based on the implementation at:
        https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/blob/master/1-zbigniewwojna/main.py#L412
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
from skimage import io, measure

from aitlas.base import BaseDataset
from aitlas.datasets.schemas import SpaceNet6DatasetSchema
from aitlas.utils.utils import parse_img_id


class SpaceNet6Dataset(BaseDataset):
    """SpaceNet6 dataset."""
    schema = SpaceNet6DatasetSchema

    def __init__(self, config):
        BaseDataset.__init__(self, config)
        self.image_paths = list()
        self.mask_paths = list()
        self.orients = pd.read_csv(config.orients, index_col=0)
        self.orients["val"] = list(range(len(self.orients.index)))

    def __getitem__(self, index):
        """
        Loads the dataset item at the specified index.
        Applies the transformations to the item before returning it.

        Parameters
        ----------
            index : int
                Specifying which item to return.
        """
        # Get image paths
        image_path = self.image_paths[index]
        # Read image
        image = io.imread(image_path)
        mask = None  # placeholder, ignores the "might be referenced before assignment" warning
        # Calculate min/max x/y for the black parts
        m = np.where((image.sum(axis=2) > 0).any(1))
        y_min, y_max = np.amin(m), np.amax(m) + 1
        m = np.where((image.sum(axis=2) > 0).any(0))
        x_min, x_max = np.amin(m), np.amax(m) + 1
        # Remove black parts
        image = image[y_min: y_max, x_min: x_max]
        # Apply transformations, (should be available only for training data)
        if self.config.apply_transforms:
            # Get mask path
            mask_path = self.mask_paths[index]
            # Read mask
            mask = io.imread(mask_path)
            # Remove black parts
            mask = mask[y_min: y_max, x_min: x_max]
            image, mask = self.transform({
                # data
                "image": image,
                "mask": mask,
                # crop size
                "crop_size": self.config.crop_size,
                # transform probabilities
                "rot_prob": self.config.rot_prob,
                "flip_lr_prob": self.config.flip_lr_prob,
            })
        # Extract direction, strip and coordinates from image
        direction, strip, coordinate = parse_img_id(image_path, self.orients)
        if direction.item():
            image = np.fliplr(np.flipud(image))
            if self.config.apply_transforms:
                mask = np.fliplr(np.flipud(mask))
        image = (image - np.array([28.62501827, 36.09922463, 33.84483687, 26.21196667])) / np.array(
            [8.41487376, 8.26645475, 8.32328472, 8.63668993])
        # Transpose image
        image = torch.from_numpy(image.transpose((2, 0, 1)).copy()).float()
        # Reorder bands
        image = image[[0, 3, 1, 2]]
        if self.config.apply_transforms:
            weights = np.ones_like(mask[:, :, :1], dtype=float)
            region_labels, region_count = measure.label(mask[:, :, 0], background=0, connectivity=1,
                                                        return_num=True)
            region_properties = measure.regionprops(region_labels)
            for bl in range(region_count):
                weights[region_labels == bl + 1] = 1024.0 / region_properties[bl].area
            mask[:, :, :3] = (mask[:, :, :3] > 1) * 1
            weights = torch.from_numpy(weights.transpose((2, 0, 1)).copy()).float()
            mask = torch.from_numpy(mask.transpose((2, 0, 1)).copy()).float()
            rgb = torch.Tensor([0])
        else:
            mask = rgb = weights = region_count = torch.Tensor([0])
        return {
            "image": image,
            "mask": mask,
            "rgb": rgb,
            "strip": strip,
            "direction": direction,
            "coordinate": coordinate,
            "image_path": image_path,
            "ymin": y_min,
            "xmin": x_min,
            "b_count": region_count,
            "weights": weights
        }

    def __len__(self):
        return len(self.image_paths)

    def load_directory(self, directory):
        """Loads the *.tif images from the specified directory."""
        self.image_paths = glob.glob(os.path.join(directory, "*.tif"))
        self.mask_paths = None

    def load_other_folds(self, fold_path, fold, root_directory, segmentation_directory):
        """Loads all images (and masks) except the ones from this fold."""
        df = pd.read_csv(fold_path)
        self.image_paths = [os.path.join(root_directory, "SAR-Intensity", os.path.basename(x)) for x in
                            df[np.logical_or(df["fold"] > (fold % 10) + 1, df["fold"] < (fold % 10) - 1)]["sar"].values]
        self.mask_paths = [os.path.join(segmentation_directory, os.path.basename(x)) for x in
                           df[np.logical_or(df["fold"] > (fold % 10) + 1, df["fold"] < (fold % 10) - 1)]["segm"].values]

    def load_fold(self, fold_path, fold, root_directory):
        """Loads the images from this fold."""
        df = pd.read_csv(fold_path)
        self.image_paths = [os.path.join(root_directory, "SAR-Intensity", os.path.basename(x)) for x in
                            df[df["fold"] == (fold % 10)]["sar"].values]
        self.mask_paths = None

    def labels(self):
        pass
