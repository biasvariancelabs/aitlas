"""
Notes
-----
    Based on the implementation at:
        https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/blob/master/1-zbigniewwojna/main.py#L412
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from skimage import io, measure

from aitlas.base import BaseDataset
# TODO: Functions: [_blend, _grayscale, saturation, brightness, contrast, parse_img_id] should be moved to utils
from aitlas.datasets.schemas import SpaceNet6DatasetSchema


def _blend(img1, img2, alpha):
    """Blends two images together."""
    return img1 * alpha + (1 - alpha) * img2


_alpha = np.asarray([0.25, 0.25, 0.25, 0.25]).reshape((1, 1, 4))


def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 4)
    return _blend(img, gs, alpha)


def parse_img_id(file_path, orients):
    """Parses direction, strip and coord components from a SpaceNet6 filepath."""
    file_name = file_path.split('/')[-1]
    strip_name = '_'.join(file_name.split('_')[-4:-2])
    direction = int(orients.loc[strip_name]['direction'])
    direction = torch.from_numpy(np.reshape(np.asarray([direction]), (1, 1, 1))).float()
    val = int(orients.loc[strip_name]['val'])
    strip = torch.Tensor(np.zeros((len(orients.index), 1, 1))).float()
    strip[val] = 1
    coord = np.asarray([orients.loc[strip_name]['coord_y']])
    coord = torch.from_numpy(np.reshape(coord, (1, 1, 1))).float() - 0.5
    return direction, strip, coord


class SpaceNet6Dataset(BaseDataset):
    """
    SpaceNet6 dataset.
    """
    schema = SpaceNet6DatasetSchema

    def __init__(self, config):
        BaseDataset.__init__(self, config)
        self.image_paths = list()
        self.mask_paths = list()
        self.orients = pd.read_csv(config.rot_out_path, index_col=0)
        self.orients['val'] = list(range(len(self.orients.index)))

    def __getitem__(self, index):
        """
        Loads the dataset item at the specified index.
        Applies the transformations to the item before returning it.

        Parameters
        ----------
            index : int
                Specifying which item to return.
        """
        image, mask = None, None
        image_path = os.path.join(self.image_paths[index])
        mask_path = os.path.join(self.mask_paths[index])
        if self.config.color:
            rgb_path = self.config.root_directory + "/PS-RGB"
            print(f"RGB path : {rgb_path}")
            print(f"Image path: {os.path.basename(self.image_paths[index]).replace('SAR-Intensity', 'PS-RGB')}")
            image_path = os.path.join(rgb_path,
                                      os.path.basename(self.image_paths[index])
                                      .replace('SAR-Intensity', 'PS-RGB')
                                      )
            image = io.imread(image_path)
            image = image[:, :, [2, 0, 0, 1]]
        else:
            image = io.imread(image_path)
        m = np.where((image.sum(axis=2) > 0).any(1))
        y_min, y_max = np.amin(m), np.amax(m) + 1
        m = np.where((image.sum(axis=2) > 0).any(0))
        x_min, x_max = np.amin(m), np.amax(m) + 1
        image = image[y_min: y_max, x_min: x_max]
        if self.config.train:
            mask = io.imread(mask_path)
            mask = mask[y_min: y_max, x_min: x_max]
            ######################################################
            image, mask = self.transform({
                # data
                "image": image,
                "mask": mask,
                # crop size
                "crop_size": self.config.crop_size,
                # transform probabilities
                "rot_prob": self.config.rot_prob,
                "scale_prob": self.config.scale_prob,
                "color_aug_prob": self.config.color_aug_prob,
                "gamma_aug_prob": self.config.gamma_aug_prob,
                "gauss_aug_prob": self.config.gauss_aug_prob,
                "elastic_aug_prob": self.config.elastic_aug_prob,
                "flipud_prob": self.config.flipud_prob,
                "fliplr_prob": self.config.fliplr_prob,
                "rot90_prob": self.config.rot90_prob,
                "channel_swap_prob": self.config.channel_swap_prob
            })
            ############################################################
        else: # val, do crop to same size
            mask = io.imread(mask_path)
            mask = mask[y_min: y_max, x_min: x_max]
        direction, strip, coordinate = parse_img_id(image_path, self.orients)
        if direction.item():
            image = np.fliplr(np.flipud(image))
            if self.config.train:
                mask = np.fliplr(np.flipud(mask))
        if self.config.color:
            image = (image - np.array([93.41131901, 97.27417209, 97.27417209, 102.25152583])) / np.array(
                [38.8338671, 41.6705231, 41.6705231, 37.34136047])
        else:
            image = (image - np.array([28.62501827, 36.09922463, 33.84483687, 26.21196667])) / np.array(
                [8.41487376, 8.26645475, 8.32328472, 8.63668993])
        image = torch.from_numpy(image.transpose((2, 0, 1)).copy()).float()
        if self.config.reorder_bands == 1:
            image = image[[2, 3, 0, 1]]
        elif self.config.reorder_bands == 2:
            image = image[[1, 3, 0, 2]]
        elif self.config.reorder_bands == 3:
            image = image[[0, 3, 1, 2]]
        if self.config.train:
            weights = np.ones_like(mask[:, :, :1], dtype=float)
            region_labels, region_count = measure.label(mask[:, :, 0], background=0, connectivity=1,
                                                        return_num=True)
            region_properties = measure.regionprops(region_labels)
            for bl in range(region_count):
                if region_properties[bl].area < self.config.train_min_building_size:
                    mask[:, :, 0][region_labels == bl + 1] = 0
                    mask[:, :, 1][region_labels == bl + 1] = 0
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

    def load_training_image_filenames(self, fold_path, fold, root_directory, segmentation_directory):
        """TODO: Documentation"""
        df = pd.read_csv(fold_path)
        self.image_paths = [os.path.join(root_directory, 'SAR-Intensity', os.path.basename(x)) for x in
                            df[np.logical_or(df['fold'] > (fold % 10) + 1, df['fold'] < (fold % 10) - 1)]['sar'].values]
        self.mask_paths = [os.path.join(segmentation_directory, os.path.basename(x)) for x in
                           df[np.logical_or(df['fold'] > (fold % 10) + 1, df['fold'] < (fold % 10) - 1)]['segm'].values]

    def load_val_image_filenames(self, fold_path, fold, root_directory, segmentation_directory):
        """TODO: Documentation"""
        df = pd.read_csv(fold_path)
        self.image_paths = [os.path.join(root_directory, 'SAR-Intensity', os.path.basename(x)) for x in
                            df[df['fold'] == (fold % 10)]['sar'].values]
        self.mask_paths = [os.path.join(segmentation_directory, os.path.basename(x)) for x in
                           df[np.logical_or(df['fold'] > (fold % 10) + 1, df['fold'] < (fold % 10) - 1)]['segm'].values]

    def labels(self):
        raise NotImplementedError
