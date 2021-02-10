import copy
import glob
import os
import shutil
import time
from functools import partial
from math import ceil
from multiprocessing import Pool, Queue, Process

import cv2
import gdal
import numpy as np
import pandas as pd
import solaris
import torch
import torch.nn.functional as F
from rasterio import features
from shapely.geometry import shape
from shapely.wkt import dumps
from skimage import io, measure
from skimage.morphology import watershed
from torch.utils.data import DataLoader
from tqdm import tqdm

from aitlas.base import BaseTask


class FocalLoss2d(torch.nn.Module):
    def __init__(self, gamma=2, ignore_index=255, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, outputs, targets, weights=1.0):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        weights = weights.contiguous()
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        weights = weights.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, self.eps, 1. - self.eps)
        targets = torch.clamp(targets, self.eps, 1. - self.eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return ((-(1. - pt) ** self.gamma * torch.log(pt)) * weights).mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, eps=1e-6):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.eps = eps

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        batch_size = outputs.size()[0]
        if not self.per_image:
            batch_size = 1
        dice_target = targets.contiguous().view(batch_size, -1).float()
        dice_output = outputs.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + self.eps
        loss = (1 - (2 * intersection + self.eps) / union).mean()
        return loss


def test_post_process(prediction_directory, prediction_csv, **kwargs):
    np.seterr(over='ignore')
    source_files = sorted(glob.glob(os.path.join(prediction_directory, '*')))
    with Pool() as pool:
        proposals = [p for p in tqdm(pool.imap_unordered(partial(test_post_process_single, **kwargs), source_files),
                                     total=len(source_files))]
    pd.concat(proposals).to_csv(prediction_csv, index=False)


def test_post_process_single(sourcefile, watershed_line=True, conn=2, polygon_buffer=0.5, tolerance=0.5,
                             seed_msk_th=0.75, area_th_for_seed=110, prediction_threshold=0.5, area_th=80,
                             contact_weight=1.0, edge_weight=0.0, seed_contact_weight=1.0, seed_edge_weight=1.0):
    mask = gdal.Open(sourcefile).ReadAsArray()  # logits
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
    mask = measure.label(mask, connectivity=conn, background=0).astype('uint8')
    polygon_generator = features.shapes(mask, mask)
    polygons = []
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(polygon_buffer)
        if p.area >= area_th:
            p = dumps(p.simplify(tolerance=tolerance), rounding_precision=0)
            polygons.append(p)
    tile_name = '_'.join(os.path.splitext(os.path.basename(sourcefile))[0].split('_')[-4:])
    csv_addition = pd.DataFrame({
        'ImageId': tile_name,
        'BuildingId': range(len(polygons)),
        'PolygonWKT_Pix': polygons,
        'Confidence': 1
    })
    return csv_addition


def evaluation(prediction_csv, gt_csv):
    evaluator = solaris.eval.base.Evaluator(gt_csv)
    evaluator.load_proposal(prediction_csv, proposalCSV=True, conf_field_list=[])
    report = evaluator.eval_iou_spacenet_csv(miniou=0.5, min_area=80)
    tp = 0
    fp = 0
    fn = 0
    for entry in report:
        tp += entry['TruePos']
        fp += entry['FalsePos']
        fn += entry['FalseNeg']
    f1score = (2 * tp) / (2 * tp + fp + fn)
    print('Validation F1 {} tp {} fp {} fn {}'.format(f1score, tp, fp, fn))
    return f1score


class SpaceNet6TrainAndEvaluateTask(BaseTask):
    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
                the model to train and evaluate
            config : Config
                the configuration for this task
        """
        BaseTask.__init__(self, model, config)

    def run(self):
        # Initialize metrics
        # TODO: refactor this as a loop
        fold = 0
        dice_loss = DiceLoss(eps=self.config.loss_eps)  # .cuda()
        focal_loss = FocalLoss2d(gamma=self.config.focal_gamma, eps=self.config.loss_eps)  # .cuda()
        # Create datasets and data loaders
        train_dataset = self.create_dataset(self.config.train_dataset_config)
        # Load filenames
        train_dataset.load_train_image_filenames(self.config.fold_path, fold, self.config.root_directory,
                                                 self.config.segmentation_directory)
        val_dataset = self.create_dataset(self.config.val_dataset_config)
        val_dataset.load_val_image_filenames(self.config.fold_path, fold, self.config.root_directory,
                                             self.config.segmentation_directory)
        train_data_loader = DataLoader(train_dataset, batch_size=train_dataset.batch_size,
                                       num_workers=train_dataset.num_workers, pin_memory=True,
                                       shuffle=train_dataset.shuffle)
        val_data_loader = DataLoader(val_dataset, batch_size=val_dataset.batch_size,
                                     num_workers=val_dataset.num_workers, pin_memory=True, shuffle=val_dataset.shuffle)
        # Initialize optimizer and lr scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100, 120], gamma=0.5)
        # Replaced `from apex import amp` with:
        # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
        # amp = torch.cuda.amp.GradScaler()
        queue = Queue()
        best_f1_score = -1
        # Kick off training
        for epoch in range(self.config.epochs if self.config.train else 1):
            if self.config.train:
                if epoch < self.config.warm_up_dec_epochs:
                    for m in [self.model.enc0, self.model.enc1, self.model.enc2, self.model.enc3, self.model.enc4]:
                        for p in m.parameters():
                            p.requires_grad = False
                else:
                    for m in [self.model.enc0, self.model.enc1, self.model.enc2, self.model.enc3, self.model.enc4]:
                        for p in m.parameters():
                            p.requires_grad = True
                time2 = time.time()
                iterator = tqdm(train_data_loader)
                self.model.train()
                # torch.cuda.empty_cache()
                for sample in iterator:
                    time1 = time.time()
                    load = time1 - time2
                    imgs = self.config.input_scale * sample["img"]  # .cuda(non_blocking=True)
                    # rgb = sample['rgb'].cuda(non_blocking=True)
                    strip = self.config.strip_scale * sample["strip"]  # .cuda(non_blocking=True)
                    direction = self.config.direction_scale * sample["direction"]  # .cuda(non_blocking=True)
                    coord = self.config.coord_scale * sample["coord"]  # .cuda(non_blocking=True)
                    target = sample["mask"]  # .cuda(non_blocking=True)
                    b_count = sample["b_count"]  # .cuda(non_blocking=True) / self.config.b_count_div
                    b_weights = b_count * self.config.b_count_weight + 1.0 * (1.0 - self.config.b_count_weight)
                    b_rev_size_weights = sample["weights"]  # .cuda(non_blocking=True)
                    b_rev_size_weights = b_rev_size_weights * self.config.b_rev_size_weight + 1.0 * (
                            1.0 - self.config.b_rev_size_weight)
                    weights = torch.ones(size=target.shape)  # .cuda()
                    weights[target > 0.0] *= self.config.pos_weight
                    weights[:, :1] *= b_rev_size_weights
                    weights[:, 1:2] *= b_rev_size_weights
                    for i in range(weights.shape[0]):
                        weights[i] = weights[i] * b_weights[i]
                    outputs = self.model(imgs, strip, direction, coord)
                    if isinstance(outputs, tuple):
                        output = outputs[1]
                        l0 = self.config.focal_weight * focal_loss(output[:, 0], target[:, 0],
                                                                   weights[:, 0]) + dice_loss(output[:, 0],
                                                                                              target[:, 0])
                        l1 = self.config.edge_weight * (
                                self.config.focal_weight * focal_loss(output[:, 1], target[:, 1],
                                                                      weights[:, 1]) + dice_loss(output[:, 1],
                                                                                                 target[:, 1]))
                        l2 = self.config.contact_weight * (
                                self.config.focal_weight * focal_loss(output[:, 2], target[:, 2],
                                                                      weights[:, 2]) + dice_loss(output[:, 2],
                                                                                                 target[:, 2]))
                        l_height = 0
                        lrgb = 0
                        loss_aux = l0 + l1 + l2 + l_height + lrgb
                    else:
                        loss_aux = 0
                        output = outputs
                    l0 = self.config.focal_weight * focal_loss(output[:, 0], target[:, 0], weights[:, 0]) + dice_loss(
                        output[:, 0], target[:, 0])
                    l1 = self.config.edge_weight * (self.config.focal_weight * focal_loss(output[:, 1], target[:, 1],
                                                                                          weights[:, 1]) + dice_loss(
                        output[:, 1], target[:, 1]))
                    l2 = self.config.contact_weight * (self.config.focal_weight * focal_loss(output[:, 2], target[:, 2],
                                                                                             weights[:, 2]) + dice_loss(
                        output[:, 2], target[:, 2]))
                    l_height = 0
                    lrgb = 0
                    loss = l0 + l1 + l2 + l_height + lrgb + self.config.aux_scale * loss_aux
                    optimizer.zero_grad()
                    if self.config.apex:
                        with amp.scale(loss) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.step(optimizer), self.config.clip_grad_norm_value)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm_value)
                    optimizer.step()
                    time2 = time.time()
                    proc = time2 - time1
                    iterator.set_description(
                        "epoch: {}; lr {:.7f}; Loss {:.4f} l0 {:.4f} l1 {:.4f} l2 {:.4f} lrgb {:.4f} l_height {:.4f} load time {:.4f} proc time {:.4f}".format(
                            epoch, scheduler.get_lr()[-1], loss, l0, l1, l2, lrgb, l_height, load, proc))
                scheduler.step()
                torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()},
                           os.path.join(self.config.model_directory, self.config.snapshot_last))
                # torch.cuda.empty_cache()
            if self.config.val and epoch > self.config.start_val_epoch:
                t.join()
                best_f1_score = max(best_f1_score, queue.get())
            if (self.config.val and epoch >= self.config.start_val_epoch) or self.config.test:
                print('Validation starts')
                shutil.rmtree(self.config.pred_folder, ignore_errors=True)
                os.makedirs(self.config.pred_folder, exist_ok=True)
                self.model.eval()
                # torch.cuda.empty_cache()
                with torch.no_grad():
                    for _, sample in enumerate(tqdm(val_data_loader)):
                        # mask = sample["mask"].numpy()
                        imgs = self.config.input_scale * sample["img"]  # .cuda(non_blocking=True)
                        ymin, xmin = sample['ymin'].item(), sample['xmin'].item()
                        strip = sample["strip"]  # .cuda(non_blocking=True)
                        direction = sample["direction"]  # .cuda(non_blocking=True)
                        coord = sample["coord"]  # .cuda(non_blocking=True)
                        _, _, h, w = imgs.shape
                        scales = [0.8, 1.0, 1.5]
                        flips = [lambda x: x, lambda x: torch.flip(x, (3,))]
                        rots = [(lambda x: torch.rot90(x, i, (2, 3))) for i in range(0, 1)]
                        rots2 = [(lambda x: torch.rot90(x, 4 - i, (2, 3))) for i in range(0, 1)]
                        oos = torch.zeros((imgs.shape[0], 6, imgs.shape[2], imgs.shape[3])).cuda()
                        for sc in scales:
                            im = F.interpolate(imgs, size=(ceil(h * sc / 32) * 32, ceil(w * sc / 32) * 32),
                                               mode='bilinear', align_corners=True)
                            for fl in flips:
                                for i, rot in enumerate(rots):
                                    o = self.model(rot(fl(im)), self.config.strip_scale * strip,
                                                   self.config.direction_scale * direction,
                                                   self.config.coord_scale * coord)
                                    if isinstance(o, tuple):
                                        o = o[0]
                                    oos += F.interpolate(fl(rots2[i](o)), size=(h, w), mode='bilinear',
                                                         align_corners=True)
                        o = oos / (len(scales) * len(flips) * len(rots))
                        o = np.moveaxis(o.cpu().data.numpy(), 1, 3)
                        for i in range(len(o)):
                            img = o[i][:, :, :3]
                            if direction[i].item():
                                img = np.fliplr(np.flipud(img))
                            img = cv2.copyMakeBorder(img, ymin, 900 - h - ymin, xmin, 900 - w - xmin,
                                                     cv2.BORDER_CONSTANT, 0.0)
                            io.imsave(os.path.join(self.config.pred_folder, os.path.split(sample['img_name'][i])[1]),
                                      img)
                # torch.cuda.empty_cache()
            if self.config.val and epoch >= self.config.start_val_epoch:
                to_save = {k: copy.deepcopy(v.cpu()) for k, v in self.model.state_dict().items()}

                def new_thread():
                    test_post_process(self.config.pred_folder, self.config.pred_csv)
                    val_f1 = evaluation(self.config.pred_csv, self.config.gt_csv)
                    print('\n    Validation loss at epoch {}: {:.5f}, best {}\n'.format(epoch, val_f1,
                                                                                        max(val_f1, best_f1_score)))
                    if best_f1_score < val_f1 and self.config.train:
                        torch.save({'epoch': epoch, 'state_dict': to_save, 'best_score': val_f1},
                                   os.path.join(self.config.model_directory, self.config.snapshot_best))
                    queue.put(val_f1)

                t = Process(target=new_thread)
                t.start()
