import glob
import os
import warnings
from functools import partial
from multiprocessing.pool import Pool

import cv2
import gdal
import math
import numpy as np
import pandas as pd
from shapely.wkt import loads
from skimage import io
from skimage.morphology import square, erosion, dilation, watershed
from tqdm import tqdm

from aitlas.base import BaseTask
from aitlas.tasks.schemas import SpaceNet6PreprocessTaskSchema

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


def process_image(image_path, segmentation_directory, edge_width, contact_width, gt_buildings_csv):
    gt_buildings = pd.read_csv(gt_buildings_csv)
    img_name = os.path.basename(image_path)
    values = gt_buildings[(gt_buildings['ImageId'] == '_'.join(img_name.split('_')[-4:])[:-4])][
        ['TileBuildingId', 'PolygonWKT_Pix', 'Mean_Building_Height']].values
    labels = np.zeros((900, 900), dtype='uint16')
    heights = np.zeros((900, 900), dtype='float')
    cur_lbl = 0
    for i in range(values.shape[0]):
        poly = loads(values[i, 1])
        if not poly.is_empty:
            cur_lbl += 1
            msk = polygon_to_mask(poly, (900, 900))
            labels[msk > 0] = cur_lbl
            if values[i, 2] == values[i, 2]:
                heights[msk > 0] = values[i, 2]
    msk = np.zeros((900, 900, 3), dtype='uint8')
    if cur_lbl > 0:
        footprint_msk = labels > 0
        border_msk = np.zeros_like(labels, dtype='bool')
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
        contact_msk = np.zeros_like(labels, dtype='bool')
        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                if labels[y0, x0] == 0:
                    sz = 3
                else:
                    sz = 1
                unique = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                                   max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
                if len(unique[unique > 0]) > 1:
                    contact_msk[y0, x0] = True
        msk = np.stack((255 * footprint_msk, 255 * border_msk, 255 * contact_msk)).astype('uint8')
        msk = np.rollaxis(msk, 0, 3)
    io.imsave(os.path.join(segmentation_directory, img_name), msk)


class SpaceNet6PreprocessTask(BaseTask):
    """Preprocessing step, creates segmentation masks."""
    schema = SpaceNet6PreprocessTaskSchema  # set up the task schema

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
                not needed, a placeholder variable
            config : Config
                the configuration for this task
        """
        BaseTask.__init__(self, model, config)

    def run(self):
        """Implements the logic of this task."""
        gt_buildings_csv = os.path.join(self.config.root_directory,
                                        'SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')
        gt_buildings = pd.read_csv(gt_buildings_csv)
        os.makedirs(self.config.segmentation_directory, exist_ok=True)
        # Walk the directory and save the filenames in it
        sar_image_paths = glob.glob(os.path.join(self.config.root_directory, 'SAR-Intensity', '*.tif'))
        # Run processing
        with Pool(self.config.num_threads) as pool:
            for _ in tqdm(pool.imap_unordered(
                    partial(process_image,
                            segmentation_directory=self.config.segmentation_directory,
                            edge_width=self.config.edge_width,
                            contact_width=self.config.contact_width,
                            gt_buildings_csv=gt_buildings_csv),
                    sar_image_paths)):
                pass
        rotations = pd.read_csv(self.config.rotation_input_path, sep=' ', index_col=0, names=['strip', 'direction'],
                                header=None)
        rotations['sum_y'] = 0.0
        rotations['ctr_y'] = 0.0
        df_fold = pd.DataFrame(columns=['ImageId', 'sar', 'segm', 'rotation', 'x', 'y', 'fold'])
        l_edge = 591640
        r_edge = 596160
        num_groups = 10
        for sar_path in tqdm(sar_image_paths):
            ImageId = '_'.join(os.path.splitext(os.path.basename(sar_path))[0].split('_')[-4:])
            strip_name = '_'.join(ImageId.split('_')[-4:-2])
            rotation = rotations.loc[strip_name]['direction'].squeeze()
            tr = gdal.Open(sar_path).GetGeoTransform()
            rotations.loc[strip_name, 'sum_y'] += tr[3]
            rotations.loc[strip_name, 'ctr_y'] += 1
            fold_no = min(num_groups - 1, max(0, math.floor((tr[0] - l_edge) / (r_edge - l_edge) * num_groups)))
            segmentation_path = os.path.join(self.config.segmentation_directory, os.path.basename(sar_path))
            df_fold = df_fold.append({
                'ImageId': ImageId,
                'sar': sar_path,
                'segm': segmentation_path,
                'rotation': rotation,
                'x': tr[0],
                'y': tr[3],
                'fold': fold_no
            }, ignore_index=True)
        df_fold.to_csv(os.path.join(self.config.fold_directory, "folds.csv"), index=False)
        for i in range(num_groups):
            print('%i: %i' % (i, len(df_fold[df_fold['fold'] == i])))
            img_ids = df_fold[df_fold['fold'] == i]['ImageId'].values
            gt_buildings[gt_buildings.ImageId.isin(img_ids)].to_csv(
                os.path.join(self.config.fold_directory, "gt_fold{0}.csv").format(i), index=False)
        rotations['mean_y'] = rotations['sum_y'] / rotations['ctr_y']
        rotations['coord_y'] = (((rotations['mean_y'] - 5746153.106161971) / 11000) + 0.2)
        rotations.to_csv(self.config.rotation_output_path, index=True)
