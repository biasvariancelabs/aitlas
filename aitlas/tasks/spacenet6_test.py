import glob
import os
import shutil
from functools import partial
from math import ceil
from multiprocessing import Pool

import cv2
import gdal
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rasterio import features
from shapely.geometry import shape
from shapely.wkt import dumps
from skimage import io, measure
from skimage.segmentation import watershed
from torch.utils.data import DataLoader
from tqdm import tqdm

from aitlas.base import BaseTask
from aitlas.tasks.schemas import SpaceNet6TestTaskSchema


def post_process(prediction_directory, prediction_csv, **kwargs):
    np.seterr(over="ignore")
    source_files = sorted(glob.glob(os.path.join(prediction_directory, '*')))
    with Pool() as pool:
        proposals = [p for p in tqdm(pool.imap_unordered(partial(post_process_single, **kwargs), source_files),
                                     total=len(source_files))]
    pd.concat(proposals).to_csv(prediction_csv, index=False)


def post_process_single(sourcefile, watershed_line=True, conn=2, polygon_buffer=0.5, tolerance=0.5,
                        seed_msk_th=0.75, area_th_for_seed=110, prediction_threshold=0.5, area_th=80,
                        contact_weight=1.0, edge_weight=0.0, seed_contact_weight=1.0, seed_edge_weight=1.0):
    mask = gdal.Open(sourcefile).ReadAsArray()
    mask = 1.0 / (1 + np.exp(-mask))
    mask[0] = mask[0] * (1 - contact_weight * mask[2]) * (1 - edge_weight * mask[1])
    seed_msk = mask[0] * (1 - seed_contact_weight * mask[2]) * (1 - seed_edge_weight * mask[1])
    seed_msk = measure.label((seed_msk > seed_msk_th), connectivity=conn, background=0)
    props = measure.regionprops(seed_msk)
    for i in range(len(props)):
        if props[i].area < area_th_for_seed:
            seed_msk[seed_msk == i + 1] = 0
    seed_msk = measure.label(seed_msk, connectivity=conn, background=0)
    mask = watershed(-mask[0], seed_msk, mask=(mask[0] > prediction_threshold), watershed_line=watershed_line)
    mask = measure.label(mask, connectivity=conn, background=0).astype("uint8")
    polygon_generator = features.shapes(mask, mask)
    polygons = []
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(polygon_buffer)
        if p.area >= area_th:
            p = dumps(p.simplify(tolerance=tolerance), rounding_precision=0)
            polygons.append(p)
    tile_name = "_".join(os.path.splitext(os.path.basename(sourcefile))[0].split("_")[-4:])
    csv_addition = pd.DataFrame({
        "ImageId": tile_name,
        "BuildingId": range(len(polygons)),
        "PolygonWKT_Pix": polygons,
        "Confidence": 1
    })
    return csv_addition


class SpaceNet6TestTask(BaseTask):
    schema = SpaceNet6TestTaskSchema  # set up the task schema

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
                the model to be used for testing (the best model from the train and evaluate task)
            config : Config
                the configuration for this task
        """
        BaseTask.__init__(self, model, config)
        # load model, best from train and evaluate task
        self.model.load_state_dict(torch.load(config.model_path)["state_dict"])

    def run(self):
        """Implements the logic of this task."""
        folds = [0, 3, 6, 9, 1, 2, 7, 8]
        for fold in folds:
            pred_folder = self.config.pred_folder.format(fold)
            # Enforce a clean do-over everytime by re-creating the destination prediction directory
            shutil.rmtree(pred_folder, ignore_errors=True)
            os.makedirs(pred_folder, exist_ok=True)
            # Create and load data set
            dataset = self.create_dataset(self.config.dataset_config)
            dataset.load_fold(self.config.fold_path, fold, self.config.root_directory)
            data_loader = DataLoader(dataset, batch_size=dataset.batch_size,
                                     num_workers=dataset.num_workers, pin_memory=True,
                                     shuffle=dataset.shuffle)
            # Set model to eval mode and clean cuda cache
            self.model.eval()
            self.model.cuda()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for sample in tqdm(data_loader):
                    images = sample["image"].cuda(non_blocking=True)
                    ymin, xmin = sample["ymin"].item(), sample["xmin"].item()
                    strip = sample["strip"].cuda(non_blocking=True)
                    direction = sample["direction"].cuda(non_blocking=True)
                    coord = sample["coordinate"].cuda(non_blocking=True)
                    _, _, h, w = images.shape
                    scales = [0.8, 1.0, 1.5]
                    flips = [lambda x: x, lambda x: torch.flip(x, (3,))]
                    rots = [(lambda x: torch.rot90(x, i, (2, 3))) for i in range(0, 1)]
                    rots2 = [(lambda x: torch.rot90(x, 4 - i, (2, 3))) for i in range(0, 1)]
                    oos = torch.zeros((images.shape[0], 6, images.shape[2], images.shape[3])).cuda()
                    for sc in scales:
                        im = F.interpolate(images, size=(ceil(h * sc / 32) * 32, ceil(w * sc / 32) * 32),
                                           mode="bilinear",
                                           align_corners=True)
                        for fl in flips:
                            for i, rot in enumerate(rots):
                                o = self.model(rot(fl(im)), strip, direction, coord)
                                if isinstance(o, tuple):
                                    o = o[0]
                                oos += F.interpolate(fl(rots2[i](o)), size=(h, w), mode="bilinear", align_corners=True)
                    o = oos / (len(scales) * len(flips) * len(rots))
                    o = np.moveaxis(o.cpu().data.numpy(), 1, 3)
                    for i in range(len(o)):
                        img = o[i][:, :, :3]
                        if direction[i].item():
                            img = np.fliplr(np.flipud(img))
                        img = cv2.copyMakeBorder(img, ymin, 900 - h - ymin, xmin, 900 - w - xmin, cv2.BORDER_CONSTANT,
                                                 0.0)
                        io.imsave(os.path.join(pred_folder, os.path.split(sample["image_path"][i])[1]), img)
            torch.cuda.empty_cache()
        ################################################################################################
        # Merge everything
        shutil.rmtree(self.config.merged_pred_dir, ignore_errors=True)
        os.makedirs(self.config.merged_pred_dir, exist_ok=True)
        merge_folds = [0, 1, 2, 3, 6, 7, 8, 9]
        predictions_folders = [self.config.pred_folder.format(i) for i in merge_folds]
        for filename in tqdm(os.listdir(predictions_folders[0])):
            used_masks = list()
            for ff in predictions_folders:
                if os.path.exists(os.path.join(ff, filename)):
                    used_masks.append(io.imread(os.path.join(ff, filename)))
            # used_masks = [io.imread(os.path.join(ff, filename)) for ff in predictions_folders]
            mask = np.zeros_like(used_masks[0], dtype="float")
            for used_mask in used_masks:
                mask += used_mask.astype("float") / len(used_masks)
            io.imsave(os.path.join(self.config.merged_pred_dir, filename), mask)
        post_process(self.config.merged_pred_dir, self.config.solution_file)
