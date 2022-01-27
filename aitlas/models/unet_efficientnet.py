"""
Notes
-----
    Based on the implementation at:
        https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/blob/master/1-zbigniewwojna/main.py#L178
"""
import copy
import glob
import os
import shutil
from functools import partial
from math import ceil
from multiprocessing import Pool, Queue

import cv2

try:
    import gdal
except ModuleNotFoundError as err:
    from osgeo import gdal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from geffnet.conv2d_layers import select_conv2d
from geffnet.efficientnet_builder import (
    BN_EPS_TF_DEFAULT,
    EfficientNetBuilder,
    decode_arch_def,
    initialize_weight_default,
    initialize_weight_goog,
    resolve_bn_args,
    round_channels,
)
from rasterio import features
from shapely.geometry import shape
from shapely.wkt import dumps
from skimage import io, measure
from skimage.segmentation import watershed
from solaris.eval.base import Evaluator
from torch.hub import load_state_dict_from_url
from tqdm import tqdm

from ..base import BaseSegmentationClassifier
from ..datasets import SpaceNet6Dataset
from ..models.schemas import UNetEfficientNetModelSchema


def post_process(prediction_directory, prediction_csv):
    np.seterr(over="ignore")
    source_files = sorted(glob.glob(os.path.join(prediction_directory, "*")))
    with Pool() as pool:
        proposals = [
            p
            for p in tqdm(
                pool.imap_unordered(partial(post_process_single), source_files),
                total=len(source_files),
            )
        ]
    if len(proposals) > 0:
        pd.concat(proposals).to_csv(prediction_csv, index=False)


def post_process_single(
    sourcefile,
    watershed_line=True,
    conn=2,
    polygon_buffer=0.5,
    tolerance=0.5,
    seed_msk_th=0.75,
    area_th_for_seed=110,
    prediction_threshold=0.5,
    area_th=80,
    contact_weight=1.0,
    edge_weight=0.0,
    seed_contact_weight=1.0,
    seed_edge_weight=1.0,
):
    mask = gdal.Open(sourcefile).ReadAsArray()
    mask = 1.0 / (1 + np.exp(-mask))
    mask[0] = mask[0] * (1 - contact_weight * mask[2]) * (1 - edge_weight * mask[1])
    seed_msk = (
        mask[0] * (1 - seed_contact_weight * mask[2]) * (1 - seed_edge_weight * mask[1])
    )
    seed_msk = measure.label((seed_msk > seed_msk_th), connectivity=conn, background=0)
    props = measure.regionprops(seed_msk)
    for i in range(len(props)):
        if props[i].area < area_th_for_seed:
            seed_msk[seed_msk == i + 1] = 0
    seed_msk = measure.label(seed_msk, connectivity=conn, background=0)
    mask = watershed(
        -mask[0],
        seed_msk,
        mask=(mask[0] > prediction_threshold),
        watershed_line=watershed_line,
    )
    mask = measure.label(mask, connectivity=conn, background=0).astype("uint8")
    polygon_generator = features.shapes(mask, mask)
    polygons = []
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(polygon_buffer)
        if p.area >= area_th:
            p = dumps(p.simplify(tolerance=tolerance), rounding_precision=0)
            polygons.append(p)
    tile_name = "_".join(
        os.path.splitext(os.path.basename(sourcefile))[0].split("_")[-4:]
    )
    csv_addition = pd.DataFrame(
        {
            "ImageId": tile_name,
            "BuildingId": range(len(polygons)),
            "PolygonWKT_Pix": polygons,
            "Confidence": 1,
        }
    )
    return csv_addition


def evaluation(prediction_csv, gt_csv):
    evaluator = Evaluator(gt_csv)
    evaluator.load_proposal(prediction_csv, proposalCSV=True, conf_field_list=[])
    report = evaluator.eval_iou_spacenet_csv(miniou=0.5)
    tp = 0
    fp = 0
    fn = 0
    for entry in report:
        tp += entry["TruePos"]
        fp += entry["FalsePos"]
        fn += entry["FalseNeg"]
    f1score = (2 * tp) / (2 * tp + fp + fn)
    # print("Validation F1 {} tp {} fp {} fn {}".format(f1score, tp, fp, fn))
    return f1score


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=3, ignore_index=255, eps=1e-6):
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
        outputs = torch.clamp(outputs, self.eps, 1.0 - self.eps)
        targets = torch.clamp(targets, self.eps, 1.0 - self.eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return ((-((1.0 - pt) ** self.gamma) * torch.log(pt)) * weights).mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, per_image=False, eps=1e-6):
        super().__init__()
        self.register_buffer("weight", weight)
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


class GenEfficientNet(nn.Module):
    def __init__(
        self,
        block_args,
        num_classes=1000,
        in_channels=3,
        num_features=1280,
        stem_size=32,
        fix_stem=False,
        channel_multiplier=1.0,
        channel_divisor=8,
        channel_min=None,
        pad_type="",
        act_layer=nn.ReLU,
        drop_connect_rate=0.0,
        se_kwargs=None,
        norm_layer=nn.BatchNorm2d,
        norm_kwargs=None,
        weight_init="goog",
    ):
        super(GenEfficientNet, self).__init__()
        stem_size = round_channels(
            stem_size, channel_multiplier, channel_divisor, channel_min
        )
        self.conv_stem = select_conv2d(
            in_channels, stem_size, 3, stride=2, padding=pad_type
        )
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        builder = EfficientNetBuilder(
            channel_multiplier,
            channel_divisor,
            channel_min,
            pad_type,
            act_layer,
            se_kwargs,
            norm_layer,
            norm_kwargs,
            drop_connect_rate,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.conv_head = select_conv2d(
            builder.in_chs, num_features, 1, padding=pad_type
        )
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        for module in self.modules():
            if weight_init == "goog":
                initialize_weight_goog(module)
            else:
                initialize_weight_default(module)


class UNetEfficientNet(BaseSegmentationClassifier):
    schema = UNetEfficientNetModelSchema

    def __init__(self, config):
        """
        Parameters
        ----------
            config : Config
                the configuration for this model
        """
        super().__init__(config)
        dec_ch = [32, 64, 128, 256, 1024]
        # Placeholder variables to avoid "might be referenced before assignment" warnings
        enc_ch = list()
        depth_multiplier, channel_multiplier = 1, 1
        url = ""
        if self.config.net == "b4":
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth"
            enc_ch = [24, 32, 56, 160, 1792]
        if self.config.net == "b5":
            channel_multiplier = 1.6
            depth_multiplier = 2.2
            url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth"
            enc_ch = [24, 40, 64, 176, 2048]
        if self.config.net == "b6":
            channel_multiplier = 1.8
            depth_multiplier = 2.6
            url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth"
            enc_ch = [32, 40, 72, 200, 2304]
        if self.config.net == "b7":
            channel_multiplier = 2.0
            depth_multiplier = 3.1
            url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth"
            enc_ch = [32, 48, 80, 224, 2560]
        if self.config.net == "l2":
            channel_multiplier = 4.3
            depth_multiplier = 5.3
            url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth"
            enc_ch = [72, 104, 176, 480, 5504]
        if self.config.stride == 16:
            dec_ch[4] = enc_ch[4]
        elif self.config.stride == 8:
            dec_ch[3] = enc_ch[4]

        def mod(cin, cout, k=3):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, padding=k // 2), nn.ReLU(inplace=True)
            )

        self.model.bot0extra = mod(206, enc_ch[4])
        self.model.bot1extra = mod(206, dec_ch[4])
        self.model.bot2extra = mod(206, dec_ch[3])
        self.model.bot3extra = mod(206, dec_ch[2])
        self.model.bot4extra = mod(206, dec_ch[1])
        self.model.bot5extra = mod(206, 6)

        self.model.dec0 = mod(enc_ch[4], dec_ch[4])
        self.model.dec1 = mod(dec_ch[4], dec_ch[3])
        self.model.dec2 = mod(dec_ch[3], dec_ch[2])
        self.model.dec3 = mod(dec_ch[2], dec_ch[1])
        self.model.dec4 = mod(dec_ch[1], dec_ch[0])

        self.model.bot0 = mod(enc_ch[3] + dec_ch[4], dec_ch[4])
        self.model.bot1 = mod(enc_ch[2] + dec_ch[3], dec_ch[3])
        self.model.bot2 = mod(enc_ch[1] + dec_ch[2], dec_ch[2])
        self.model.bot3 = mod(enc_ch[0] + dec_ch[1], dec_ch[1])

        self.model.up = nn.Upsample(scale_factor=2)
        self.model.upps = nn.PixelShuffle(upscale_factor=2)
        self.model.final = nn.Conv2d(dec_ch[0], 6, 1)
        self._initialize_weights()
        arch_def = [
            ["ds_r1_k3_s1_e1_c16_se0.25"],
            ["ir_r2_k3_s2_e6_c24_se0.25"],
            ["ir_r2_k5_s2_e6_c40_se0.25"],
            ["ir_r3_k3_s2_e6_c80_se0.25"],
            ["ir_r3_k5_s1_e6_c112_se0.25"],
            ["ir_r4_k5_s2_e6_c192_se0.25"],
            ["ir_r1_k3_s1_e6_c320_se0.25"],
        ]
        enc = GenEfficientNet(
            in_channels=3,
            block_args=decode_arch_def(arch_def, depth_multiplier),
            num_features=round_channels(1280, channel_multiplier, 8, None),
            stem_size=32,
            channel_multiplier=channel_multiplier,
            norm_kwargs=resolve_bn_args({"bn_eps": BN_EPS_TF_DEFAULT}),
            pad_type="same",
        )
        state_dict = load_state_dict_from_url(url)
        enc.load_state_dict(state_dict, strict=True)
        stem_size = round_channels(32, channel_multiplier, 8, None)
        conv_stem = select_conv2d(4, stem_size, 3, stride=2, padding="same")
        _w = enc.conv_stem.state_dict()
        _w["weight"] = torch.cat([_w["weight"], _w["weight"][:, 1:2]], 1)
        conv_stem.load_state_dict(_w)
        self.model.enc0 = nn.Sequential(conv_stem, enc.bn1, enc.act1, enc.blocks[0])
        self.model.enc1 = nn.Sequential(enc.blocks[1])
        self.model.enc2 = nn.Sequential(enc.blocks[2])
        self.model.enc3 = nn.Sequential(enc.blocks[3], enc.blocks[4])
        self.model.enc4 = nn.Sequential(
            enc.blocks[5], enc.blocks[6], enc.conv_head, enc.bn2, enc.act2
        )

    def forward(self, x, strip, direction, coord):
        enc0 = self.model.enc0(x)
        enc1 = self.model.enc1(enc0)
        enc2 = self.model.enc2(enc1)
        enc3 = self.model.enc3(enc2)
        enc4 = self.model.enc4(enc3)
        ex = torch.cat([strip, direction, coord], 1)
        x = enc4
        if self.config.stride == 32:
            x = self.model.dec0(self.model.up(x + self.model.bot0extra(ex)))
            x = torch.cat([x, enc3], dim=1)
            x = self.model.bot0(x)
        if self.config.stride == 32 or self.config.stride == 16:
            x = self.model.dec1(self.model.up(x + self.model.bot1extra(ex)))
            x = torch.cat([x, enc2], dim=1)
            x = self.model.bot1(x)
        x = self.model.dec2(self.model.up(x))
        x = torch.cat([x, enc1], dim=1)
        x = self.model.bot2(x)
        x = self.model.dec3(self.model.up(x))
        x = torch.cat([x, enc0], dim=1)
        x = self.model.bot3(x)
        x = self.model.dec4(self.model.up(x))
        x = self.model.final(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight.data = nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def load_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-2)

    def load_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(
            self.load_optimizer(), milestones=[80, 100, 120], gamma=0.5
        )

    def train_and_evaluate_model(
        self,
        train_dataset: SpaceNet6Dataset,
        epochs: int = 100,
        model_directory: str = None,
        save_epochs: int = 10,
        iterations_log: int = 100,
        resume_model: str = None,
        val_dataset: SpaceNet6Dataset = None,
        run_id: str = None,
        **kwargs,
    ):
        """Overridden method for training on the SpaceNet6 data set."""
        contact_weight = train_dataset.config.contact_weight
        edge_weight = train_dataset.config.edge_weight
        fold = 0
        pred_folder = train_dataset.config.pred_folder.format(fold)
        # Initialize loss functions
        dice_loss = DiceLoss().to(self.device)
        focal_loss = FocalLoss2d().to(self.device)
        # Load training data set
        train_dataset.load_other_folds(fold)
        train_data_loader = train_dataset.dataloader()
        # Load validation data set
        val_dataset.load_fold(fold)
        val_data_loader = val_dataset.dataloader()
        # Initialize optimizer and lr scheduler
        optimizer = self.load_optimizer()
        scheduler = self.load_lr_scheduler()
        best_f1_score = -1
        # Kick off training
        self.model.to(self.device)
        for epoch in range(epochs):
            iterator = tqdm(train_data_loader)
            self.model.train()
            # For each batch (i.e. sample)
            for sample in iterator:
                images = sample["image"].to(self.device)
                strip = sample["strip"].to(self.device)
                direction = sample["direction"].to(self.device)
                coord = sample["coordinate"].to(self.device)
                target = sample["mask"].to(self.device)
                building_count = sample["b_count"].to(self.device) / 8
                building_weight = building_count * 0.5 + 0.5
                weights = torch.ones(size=target.shape).to(self.device)
                weights[target > 0.0] *= 0.5
                for i in range(weights.shape[0]):
                    weights[i] = weights[i] * building_weight[i]
                output = self.forward(images, strip, direction, coord)
                if isinstance(output, tuple):
                    output = output[0]
                l0 = focal_loss(output[:, 0], target[:, 0], weights[:, 0]) + dice_loss(
                    output[:, 0], target[:, 0]
                )
                l1 = edge_weight * (
                    focal_loss(output[:, 1], target[:, 1], weights[:, 1])
                    + dice_loss(output[:, 1], target[:, 1])
                )
                l2 = contact_weight * (
                    focal_loss(output[:, 2], target[:, 2], weights[:, 2])
                    + dice_loss(output[:, 2], target[:, 2])
                )
                loss = l0 + l1 + l2
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.2)
                optimizer.step()
                iterator.set_description(
                    "epoch: {}; lr {:.5f}; loss {:.4f}".format(
                        epoch, scheduler.get_lr()[-1], loss
                    )
                )
            scheduler.step()
            torch.save(
                {"epoch": epoch, "state_dict": self.model.state_dict()},
                os.path.join(model_directory, "last_model"),
            )
            if epoch >= val_dataset.config.start_val_epoch:
                shutil.rmtree(pred_folder, ignore_errors=True)
                os.makedirs(pred_folder, exist_ok=True)
                self.model.eval()
                with torch.no_grad():
                    for sample in tqdm(val_data_loader):
                        images = sample["image"].to(self.device)
                        ymin, xmin = sample["ymin"].item(), sample["xmin"].item()
                        strip = sample["strip"].to(self.device)
                        direction = sample["direction"].to(self.device)
                        coord = sample["coordinate"].to(self.device)
                        _, _, h, w = images.shape
                        scales = [0.8, 1.0, 1.5]
                        flips = [lambda x: x, lambda x: torch.flip(x, (3,))]
                        rots = [
                            (lambda x: torch.rot90(x, i, (2, 3))) for i in range(0, 1)
                        ]
                        rots2 = [
                            (lambda x: torch.rot90(x, 4 - i, (2, 3)))
                            for i in range(0, 1)
                        ]
                        oos = torch.zeros(
                            (images.shape[0], 6, images.shape[2], images.shape[3])
                        ).to(self.device)
                        for sc in scales:
                            im = F.interpolate(
                                images,
                                size=(ceil(h * sc / 32) * 32, ceil(w * sc / 32) * 32),
                                mode="bilinear",
                                align_corners=True,
                            )
                            for fl in flips:
                                for i, rot in enumerate(rots):
                                    o = self.forward(
                                        rot(fl(im)), strip, direction, coord
                                    )
                                    if isinstance(o, tuple):
                                        o = o[0]
                                    oos += F.interpolate(
                                        fl(rots2[i](o)),
                                        size=(h, w),
                                        mode="bilinear",
                                        align_corners=True,
                                    )
                        o = oos / (len(scales) * len(flips) * len(rots))
                        o = np.moveaxis(o.cpu().data.numpy(), 1, 3)
                        for i in range(len(o)):
                            img = o[i][:, :, :3]
                            if direction[i].item():
                                img = np.fliplr(np.flipud(img))
                            img = cv2.copyMakeBorder(
                                img,
                                ymin,
                                900 - h - ymin,
                                xmin,
                                900 - w - xmin,
                                cv2.BORDER_CONSTANT,
                                0.0,
                            )
                            io.imsave(
                                os.path.join(
                                    pred_folder,
                                    os.path.split(sample["image_path"][i])[1],
                                ),
                                img,
                            )
                to_save = {
                    k: copy.deepcopy(v.cpu())
                    for k, v in self.model.state_dict().items()
                }
                pred_csv = val_dataset.config.pred_csv.format(fold)
                gt_csv = val_dataset.config.gt_csv.format(fold)
                post_process(pred_folder, pred_csv)
                val_f1 = evaluation(pred_csv, gt_csv)
                if best_f1_score < val_f1:
                    torch.save(
                        {"epoch": epoch, "state_dict": to_save, "best_score": val_f1},
                        os.path.join(model_directory, "best_model"),
                    )
                best_f1_score = max(best_f1_score, val_f1)

    def evaluate(self, dataset: SpaceNet6Dataset = None, model_path: str = None):
        # Load the model
        self.load_model(model_path)
        # evaluate model on data
        fold = 3  # [0, 6, 9, 1, 2, 7, 8]
        # load data into the data set
        dataset.load_fold(fold)
        # get test data loader
        data_loader = dataset.dataloader()
        pred_folder = dataset.config.pred_folder.format(fold)
        # Enforce a clean do-over everytime by re-creating the destination prediction directory
        shutil.rmtree(pred_folder, ignore_errors=True)
        os.makedirs(pred_folder, exist_ok=True)
        # Set model to eval mode
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(data_loader):
                images = sample["image"].to(self.device)
                ymin, xmin = sample["ymin"].item(), sample["xmin"].item()
                strip = sample["strip"].to(self.device)
                direction = sample["direction"].to(self.device)
                coord = sample["coordinate"].to(self.device)
                _, _, h, w = images.shape
                scales = [0.8, 1.0, 1.5]
                flips = [lambda x: x, lambda x: torch.flip(x, (3,))]
                rots = [(lambda x: torch.rot90(x, i, (2, 3))) for i in range(0, 1)]
                rots2 = [(lambda x: torch.rot90(x, 4 - i, (2, 3))) for i in range(0, 1)]
                oos = torch.zeros(
                    (images.shape[0], 6, images.shape[2], images.shape[3])
                ).to(self.device)
                for sc in scales:
                    im = F.interpolate(
                        images,
                        size=(ceil(h * sc / 32) * 32, ceil(w * sc / 32) * 32),
                        mode="bilinear",
                        align_corners=True,
                    )
                    for fl in flips:
                        for i, rot in enumerate(rots):
                            o = self.forward(rot(fl(im)), strip, direction, coord)
                            if isinstance(o, tuple):
                                o = o[0]
                            oos += F.interpolate(
                                fl(rots2[i](o)),
                                size=(h, w),
                                mode="bilinear",
                                align_corners=True,
                            )
                o = oos / (len(scales) * len(flips) * len(rots))
                o = np.moveaxis(o.cpu().data.numpy(), 1, 3)
                for i in range(len(o)):
                    img = o[i][:, :, :3]
                    if direction[i].item():
                        img = np.fliplr(np.flipud(img))
                    img = cv2.copyMakeBorder(
                        img,
                        ymin,
                        900 - h - ymin,
                        xmin,
                        900 - w - xmin,
                        cv2.BORDER_CONSTANT,
                        0.0,
                    )
                    io.imsave(
                        os.path.join(
                            pred_folder, os.path.split(sample["image_path"][i])[1]
                        ),
                        img,
                    )
        ################################################################################################
        # # Merge everything
        # shutil.rmtree(dataset.config.merged_pred_dir, ignore_errors=True)
        # os.makedirs(dataset.config.merged_pred_dir, exist_ok=True)
        # merge_folds = [0, 1, 2, 3, 6, 7, 8, 9]
        # predictions_folders = [dataset.config.pred_folder.format(i) for i in merge_folds]
        # for filename in tqdm(os.listdir(predictions_folders[0])):
        #     used_masks = list()
        #     for ff in predictions_folders:
        #         if os.path.exists(os.path.join(ff, filename)):
        #             used_masks.append(io.imread(os.path.join(ff, filename)))
        #     mask = np.zeros_like(used_masks[0], dtype="float")
        #     for used_mask in used_masks:
        #         mask += used_mask.astype("float") / len(used_masks)
        #     io.imsave(os.path.join(dataset.config.merged_pred_dir, filename), mask)
        # post_process(dataset.config.merged_pred_dir, dataset.config.solution_file)

    def load_model(self, file_path, optimizer=None):
        loaded = torch.load(file_path)

        missing_keys = []
        unexpected_keys = []

        metadata = getattr(loaded["state_dict"], "_metadata", None)
        state_dict = loaded["state_dict"].copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                [],
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self.model)
