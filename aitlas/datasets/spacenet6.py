"""
.. note:: Based on the implementation at: https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/blob/master/1-zbigniewwojna/main.py#L412

"""
import glob
import math
import os
import warnings
from functools import partial
from multiprocessing import Pool

import cv2

try:
    import gdal
except ModuleNotFoundError as err:
    from osgeo import gdal

import numpy as np
import pandas as pd
import torch
from shapely.wkt import loads
from skimage import io, measure
from skimage.morphology import dilation, erosion, square
from skimage.segmentation import watershed
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from aitlas.base import BaseDataset
from aitlas.datasets.schemas import SpaceNet6DatasetSchema
from aitlas.utils import parse_img_id
from aitlas.utils import image_loader


# Ignore the "low-contrast" warnings
warnings.filterwarnings("ignore")


def polygon_to_mask(poly, image_size):
    image_mask = np.zeros(image_size, np.uint8)

    def integer_coordinates(x):
        return np.array(x).round().astype(np.int32)

    exteriors = [integer_coordinates(poly.exterior.coords)]
    interiors = [integer_coordinates(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(image_mask, exteriors, 1)
    cv2.fillPoly(image_mask, interiors, 0)
    return image_mask


def process_image(
    image_path, segmentation_directory, edge_width, contact_width, gt_buildings_csv
):
    """
    Creates and saves the target (ground-truth) segmentation mask for the input image.

    :param image_path: path to the source image
    :type image_path: str
    :param segmentation_directory: path to the destination directory for the segmentation masks
    :type segmentation_directory: str
    :param edge_width: the width of the edge
    :type edge_width: int
    :param contact_width: the width of the contact
    :type contact_width: int
    :param gt_buildings_csv: path to the source ground-truth-buildings csv
    :type gt_buildings_csv: str

    """
    gt_buildings = pd.read_csv(gt_buildings_csv)
    image_name = os.path.basename(image_path)
    values = gt_buildings[
        (gt_buildings["ImageId"] == "_".join(image_name.split("_")[-4:])[:-4])
    ][["TileBuildingId", "PolygonWKT_Pix", "Mean_Building_Height"]].values
    labels = np.zeros((900, 900), dtype="uint16")
    heights = np.zeros((900, 900), dtype="float")
    cur_lbl = 0
    for i in range(values.shape[0]):
        poly = loads(values[i, 1])
        if not poly.is_empty:
            cur_lbl += 1
            msk = polygon_to_mask(poly, (900, 900))
            labels[msk > 0] = cur_lbl
            if values[i, 2] == values[i, 2]:
                heights[msk > 0] = values[i, 2]
    msk = np.zeros((900, 900, 3), dtype="uint8")
    if cur_lbl > 0:
        footprint_msk = labels > 0
        border_msk = np.zeros_like(labels, dtype="bool")
        for l in range(1, labels.max() + 1):
            tmp_lbl = labels == l
            _k = square(edge_width)
            tmp = erosion(tmp_lbl, _k)
            tmp = tmp ^ tmp_lbl
            border_msk = border_msk | tmp
        tmp = dilation(labels > 0, square(contact_width))
        tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = tmp | border_msk
        tmp = dilation(tmp, square(contact_width))
        contact_msk = np.zeros_like(labels, dtype="bool")
        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                if labels[y0, x0] == 0:
                    sz = 3
                else:
                    sz = 1
                unique = np.unique(
                    labels[
                        max(0, y0 - sz) : min(labels.shape[0], y0 + sz + 1),
                        max(0, x0 - sz) : min(labels.shape[1], x0 + sz + 1),
                    ]
                )
                if len(unique[unique > 0]) > 1:
                    contact_msk[y0, x0] = True
        msk = np.stack(
            (255 * footprint_msk, 255 * border_msk, 255 * contact_msk)
        ).astype("uint8")
        msk = np.rollaxis(msk, 0, 3)
    io.imsave(os.path.join(segmentation_directory, image_name), msk)


class SpaceNet6Dataset(BaseDataset):
    """SpaceNet6 dataset."""

    name = "SpaceNet 6"
    schema = SpaceNet6DatasetSchema
    labels = ["background","building boundary","building"]
    color_mapping = [[0,0,0],[255,255,0],[255,0,0]]

    def __init__(self, config):
        super().__init__(config)
        self.image_paths, self.mask_paths = self.load_directory()
        self.orients = pd.read_csv(self.config.orients, index_col=0)
        self.orients["val"] = list(range(len(self.orients.index)))

    def __getitem__(self, index):
        """
        Loads the dataset item at the specified index.
        Applies the transformations to the item before returning it.

        :param index : index
        :type index: int
        """
        # Get image paths
        image_path = self.image_paths[index]
        # Read image
        image = io.imread(image_path)[:,:,[0, 3, 1]]
        # Calculate min/max x/y for the black parts
        m = np.where((image.sum(axis=2) > 0).any(1))
        y_min, y_max = np.amin(m), np.amax(m) + 1
        m = np.where((image.sum(axis=2) > 0).any(0))
        x_min, x_max = np.amin(m), np.amax(m) + 1
        # Remove black parts
        image = image[y_min:y_max, x_min:x_max]
        
        # Get mask path
        mask_path = self.mask_paths[index]
        # Read mask
        mask = io.imread(mask_path)
        # Remove black parts
        mask = mask[y_min:y_max, x_min:x_max]
        # Convert 2D mask with class IDs to one-hot encoded 3D mask
        single_band_mask = np.zeros([mask.shape[0], mask.shape[1]], np.uint8)
        
        for i in np.arange(mask.shape[0]):
            for j in  np.arange(mask.shape[1]):
                if mask[i,j,0] == 0 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 0
                if mask[i,j,0] == 255 and mask[i,j,1] == 255 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 1
                if mask[i,j,0] == 255 and mask[i,j,1] == 0 and mask[i,j,2] == 0:
                    single_band_mask[i,j] = 2
        
        masks = [(single_band_mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")

        # Extract direction, strip and coordinates from image
        direction, strip, coordinate = parse_img_id(image_path, self.orients)
        if direction.item():
            image = np.fliplr(np.flipud(image))
            if self.config.transforms:
                mask = np.fliplr(np.flipud(mask))

        #normalize image for better visualization
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-6:
            image = (image - img_min) / (img_max - img_min + 1e-8)
        # Transpose image
        image = torch.from_numpy(image.transpose((2, 0, 1)))#.copy()).float()
        
        weights = np.ones_like(mask[:, :, :1], dtype=float)
        region_labels, region_count = measure.label(
            mask[:, :, 0], background=0, connectivity=1, return_num=True
        )
        region_properties = measure.regionprops(region_labels)
        for bl in range(region_count):
            weights[region_labels == bl + 1] = 1024.0 / region_properties[bl].area
        weights = torch.from_numpy(weights.transpose((2, 0, 1)).copy()).float()
        mask = torch.from_numpy(mask.transpose((2, 0, 1)).copy()).float()
        rgb = torch.Tensor([0])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        output = {
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
            "weights": weights,
        }

        return image, mask
        
    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return self.labels

    def load_directory(self):
        """Loads the *.tif images from the specified directory."""
        self.image_paths = glob.glob(os.path.join(self.config.root_directory, "*.tif"))
        self.mask_paths = glob.glob(os.path.join(self.config.segmentation_directory, "*.tif"))

        return self.image_paths, self.mask_paths

    def load_other_folds(self, fold):
        """Loads all images (and masks) except the ones from this fold."""
        df = pd.read_csv(self.config.folds_path)
        self.image_paths = [
            os.path.join(
                self.config.root_directory, "SAR-Intensity", os.path.basename(x)
            )
            for x in df[
                np.logical_or(
                    df["fold"] > (fold % 10) + 1, df["fold"] < (fold % 10) - 1
                )
            ]["sar"].values
        ]
        self.mask_paths = [
            os.path.join(self.config.segmentation_directory, os.path.basename(x))
            for x in df[
                np.logical_or(
                    df["fold"] > (fold % 10) + 1, df["fold"] < (fold % 10) - 1
                )
            ]["segm"].values
        ]

    def load_fold(self, fold):
        """Loads the images from this fold."""
        df = pd.read_csv(self.config.folds_path)
        self.image_paths = [
            os.path.join(
                self.config.root_directory, "SAR-Intensity", os.path.basename(x)
            )
            for x in df[df["fold"] == (fold % 10)]["sar"].values
        ]
        self.mask_paths = None

    def prepare(self):
        """
        Prepares the SpaceNet6 data set for model training and validation by:

        1. Creating training segmentation masks from the geojson files

        2. Splitting the data set by location, which was shown to be very important for model learning, see:
        https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/blob/master/1-zbigniewwojna/README.md
        Creates 10 splits of the data set. Each split consists of 10 folds (i.e. further splits) of which 9 are used for
        training and one for validation/testing (in essence, a cross validation procedure).
        """
        # Create destination directories if they don't exist
        os.makedirs(self.config.segmentation_directory, exist_ok=True)
        os.makedirs(self.config.folds_dir, exist_ok=True)
        # Path to the ground-truth buildings csv file
        gt_buildings_csv_filepath = os.path.join(
            self.config.root_directory,
            "SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv",
        )
        # Read gt building csv file
        gt_buildings = pd.read_csv(gt_buildings_csv_filepath)
        # Walk the raw data directory with the SAR images and save the filenames in it
        sar_image_paths = glob.glob(
            os.path.join(self.config.root_directory, "SAR-Intensity", "*.tif")
        )
        # Process each SAR image
        with Pool(self.config.num_threads) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    partial(
                        process_image,
                        segmentation_directory=self.config.segmentation_directory,
                        edge_width=self.config.edge_width,
                        contact_width=self.config.contact_width,
                        gt_buildings_csv=gt_buildings_csv_filepath,
                    ),
                    sar_image_paths,
                )
            ):
                pass
        orientations = pd.read_csv(
            filepath_or_buffer=self.config.orients,
            sep=" ",
            index_col=0,
            names=["strip", "direction"],
            header=None,
        )
        df_fold = pd.DataFrame(
            columns=["ImageId", "sar", "segm", "rotation", "x", "y", "fold"]
        )
        l_edge = 591640
        r_edge = 596160
        orientations["sum_y"] = 0.0
        orientations["ctr_y"] = 0.0
        for sar_path in tqdm(sar_image_paths):
            image_id = "_".join(
                os.path.splitext(os.path.basename(sar_path))[0].split("_")[-4:]
            )
            strip_name = "_".join(image_id.split("_")[-4:-2])
            rotation = orientations.loc[strip_name]["direction"].squeeze()
            tr = gdal.Open(sar_path).GetGeoTransform()
            orientations.loc[strip_name, "sum_y"] += tr[3]
            orientations.loc[strip_name, "ctr_y"] += 1
            fold_no = min(
                self.config.num_folds - 1,
                max(
                    0,
                    math.floor(
                        (tr[0] - l_edge) / (r_edge - l_edge) * self.config.num_folds
                    ),
                ),
            )
            segmentation_path = os.path.join(
                self.config.segmentation_directory, os.path.basename(sar_path)
            )
            df_fold = df_fold.append(
                {
                    "ImageId": image_id,
                    "sar": sar_path,
                    "segm": segmentation_path,
                    "rotation": rotation,
                    "x": tr[0],
                    "y": tr[3],
                    "fold": fold_no,
                },
                ignore_index=True,
            )
        df_fold.to_csv(os.path.join(self.config.folds_dir, "folds.csv"), index=False)
        for i in range(self.config.num_folds):
            img_ids = df_fold[df_fold["fold"] == i]["ImageId"].values
            gt_buildings[gt_buildings.ImageId.isin(img_ids)].to_csv(
                os.path.join(self.config.folds_dir, "gt_fold{}.csv").format(i),
                index=False,
            )
        orientations["mean_y"] = orientations["sum_y"] / orientations["ctr_y"]
        orientations["coord_y"] = (
            (orientations["mean_y"] - 5746153.106161971) / 11000
        ) + 0.2
        orientations.to_csv(self.config.orients_output, index=True)

    def show_image(self, index, show_title=False):
        img, mask = self[index]
        img = img.permute(1,2,0)
        mask = mask.permute(1,2,0)
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
        height_factor = math.ceil(len(self.labels)/3)
        if height_factor == 4:
            height_factor = 0.73
        elif height_factor == 2:
            height_factor = 0.80
        else:
            height_factor = 0.81
        fig.legend(handles=legend_elements, bbox_to_anchor=(0.2, height_factor, 0.6, 0.2), ncol=3, mode='expand',
                   loc='lower left', prop={'size': 12})
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_mask)
        plt.axis("off")
        fig.tight_layout()
        plt.show()
        return fig
    
    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for i in np.arange(len(self.image_paths)):
            _, mask = self.__getitem__(i)
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        return label_count

    def data_distribution_barchart(self, show_title=True):
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(data=label_count, x=label_count.index, y='Number of pixels', ax=ax)
        if show_title:
            ax.set_title(
                "Labels distribution for {}".format(self.get_name()), pad=20, fontsize=18
            )
        return fig