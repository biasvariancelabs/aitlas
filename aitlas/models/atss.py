"""ATSS: Adaptive Training Sample Selection"""

import math
import os
import re
from typing import List, Optional

import requests
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.boxes import box_area
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from ..base.object_detection import BaseObjectDetection


# -----------------------------------------------------------------------------
# Utils & Helper Functions
# -----------------------------------------------------------------------------


def box_iou(boxes1, boxes2):
    """
    From atss_core/structures/boxlist_ops.py
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0):
    """
    Standard PyTorch focal loss implementation
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


def giou_loss(pred_boxes, target_boxes, weight=None):
    """
    From atss_core/modeling/rpn/atss/loss.py
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(-1)
    target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(-1)

    pred_x2 = torch.max(pred_x1, pred_x2)
    pred_y2 = torch.max(pred_y1, pred_y2)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    x1_intersect = torch.max(pred_x1, target_x1)
    y1_intersect = torch.max(pred_y1, target_y1)
    x2_intersect = torch.min(pred_x2, target_x2)
    y2_intersect = torch.min(pred_y2, target_y2)

    area_intersect = torch.zeros_like(pred_x1)
    mask = (y2_intersect > y1_intersect) & (x2_intersect > x1_intersect)
    area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (
        y2_intersect[mask] - y1_intersect[mask]
    )

    x1_enclosing = torch.min(pred_x1, target_x1)
    y1_enclosing = torch.min(pred_y1, target_y1)
    x2_enclosing = torch.max(pred_x2, target_x2)
    y2_enclosing = torch.max(pred_y2, target_y2)
    area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

    area_union = pred_area + target_area - area_intersect + 1e-7
    ious = area_intersect / area_union
    gious = ious - (area_enclosing - area_union) / area_enclosing

    losses = 1 - gious
    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        return losses.sum()


# -----------------------------------------------------------------------------
# Core ATSS Modules
# -----------------------------------------------------------------------------


class Scale(nn.Module):
    """
    From atss_core/layers/scale.py
    """

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class BoxCoder:
    """
    From atss_core/modeling/rpn/atss/atss.py
    """

    def encode(self, gt_boxes, anchors):
        ex_widths = anchors[:, 2] - anchors[:, 0]
        ex_heights = anchors[:, 3] - anchors[:, 1]
        ex_ctr_x = anchors[:, 0] + 0.5 * ex_widths
        ex_ctr_y = anchors[:, 1] + 0.5 * ex_heights

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

        targets_dx = 10.0 * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = 10.0 * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = 5.0 * torch.log(gt_widths / ex_widths)
        targets_dh = 5.0 * torch.log(gt_heights / ex_heights)
        return torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

    def decode(self, preds, anchors):
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        dx = preds[:, 0] / 10.0
        dy = preds[:, 1] / 10.0
        dw = preds[:, 2] / 5.0
        dh = preds[:, 3] / 5.0

        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes


class AnchorGenerator(nn.Module):
    """
    From atss_core/modeling/rpn/anchor_generator.py
    """

    def __init__(self, anchor_sizes, anchor_strides):
        super().__init__()
        self.anchor_sizes = anchor_sizes
        self.anchor_strides = anchor_strides

    def forward(self, feature_maps):
        anchors = []
        for size, stride, feat in zip(self.anchor_sizes, self.anchor_strides, feature_maps):
            h, w = feat.shape[-2:]
            device = feat.device
            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
            shift_y, shift_x = (
                torch.meshgrid(shifts_y, shifts_x, indexing="ij")
                if int(torch.__version__.split(".")[1]) >= 10
                else torch.meshgrid(shifts_y, shifts_x)
            )
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1) + stride // 2
            base_anchor = torch.tensor(
                [-size / 2, -size / 2, size / 2, size / 2],
                dtype=torch.float32,
                device=device,
            )
            anchors.append(shifts + base_anchor)
        return anchors


class BackboneWithFPN(nn.Module):
    """
    Standard ResNet + FPN structured to strictly match the ATSS implementation
    independently from unstable torchvision internal extra block signatures.
    """

    def __init__(self, backbone_name, pretrained=True, in_channels=3):
        super().__init__()
        backbone = getattr(torchvision.models, backbone_name)(pretrained=pretrained)

        # Patch first conv layer if input channels != 3
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            backbone.conv1 = new_conv

        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        in_channels_list = (
            [512, 1024, 2048] if backbone_name not in ["resnet18", "resnet34"] else [128, 256, 512]
        )
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=256)

        # ATSS generates P6 and P7 on top of P5
        self.p6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        c = self.body(x)
        p = self.fpn(c)

        p5 = p["2"]
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))

        return [p["0"], p["1"], p["2"], p6, p7]


class ATSSHead(nn.Module):
    """
    From atss_core/modeling/rpn/atss/atss.py
    """

    def __init__(self, in_channels, num_classes, num_anchors=1):
        super().__init__()
        cls_tower = []
        bbox_tower = []
        for i in range(4):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))

            bbox_tower.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(
            in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for modules in [
            self.cls_tower,
            self.bbox_tower,
            self.cls_logits,
            self.bbox_pred,
            self.centerness,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # Bias Initialization for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            bbox_reg.append(self.scales[l](self.bbox_pred(box_tower)))
            centerness.append(self.centerness(box_tower))

        return logits, bbox_reg, centerness


# -----------------------------------------------------------------------------
# Main ATSS Model Architecture (from atss_core/modeling/rpn/atss/atss.py)
# -----------------------------------------------------------------------------


class ATSSModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, backbone_name="resnet50", in_channels=3):
        super().__init__()
        self.backbone = BackboneWithFPN(
            backbone_name, pretrained=pretrained, in_channels=in_channels
        )
        self.head = ATSSHead(256, num_classes)

        self.anchor_sizes = [64, 128, 256, 512, 1024]
        self.anchor_strides = [8, 16, 32, 64, 128]
        self.anchor_generator = AnchorGenerator(self.anchor_sizes, self.anchor_strides)
        self.box_coder = BoxCoder()
        self.num_classes = num_classes
        self.topk = 9

    def forward(self, images):
        feature_list = self.backbone(images)
        logits, bbox_reg, centerness = self.head(feature_list)
        anchors_per_level = self.anchor_generator(feature_list)
        return logits, bbox_reg, centerness, anchors_per_level


# -----------------------------------------------------------------------------
# Target Generation & Centerness (from atss_core/modeling/rpn/atss/loss.py)
# -----------------------------------------------------------------------------


def compute_atss_targets(anchors, gt_boxes, gt_labels, num_anchors_per_level, topk=9):
    num_gt = gt_boxes.shape[0]
    num_anchors = anchors.shape[0]

    if num_gt == 0:
        return torch.full(
            (num_anchors,), -1, dtype=torch.int64, device=anchors.device
        ), torch.zeros((num_anchors, 4), dtype=torch.float32, device=anchors.device)

    ious, _ = box_iou(anchors, gt_boxes)  # [N, M]

    gt_cx = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2.0
    gt_cy = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2.0
    gt_points = torch.stack((gt_cx, gt_cy), dim=1)  # [M, 2]

    anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2.0
    anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2.0
    anchor_points = torch.stack((anchors_cx, anchors_cy), dim=1)  # [N, 2]

    distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()  # [N, M]

    # Select candidates based on L2 distance
    candidate_idxs = []
    start_idx = 0
    for num_anchors_in_level in num_anchors_per_level:
        end_idx = start_idx + num_anchors_in_level
        distances_per_level = distances[start_idx:end_idx, :]
        topk_level = min(topk, num_anchors_in_level)
        _, topk_idxs_per_level = distances_per_level.topk(topk_level, dim=0, largest=False)
        candidate_idxs.append(topk_idxs_per_level + start_idx)
        start_idx = end_idx
    candidate_idxs = torch.cat(candidate_idxs, dim=0)  # [L*topk, M]

    # Compute statistical IoU Thresholds
    candidate_ious = ious.gather(0, candidate_idxs)  # [L*topk, M]
    iou_mean_per_gt = candidate_ious.mean(0)
    iou_std_per_gt = candidate_ious.std(0)
    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
    is_pos = candidate_ious >= iou_thresh_per_gt[None, :]  # [L*topk, M]

    # Limit positive samples' center entirely within the object
    candidate_anchors_cx = anchors_cx[candidate_idxs]  # [L*topk, M]
    candidate_anchors_cy = anchors_cy[candidate_idxs]

    l = candidate_anchors_cx - gt_boxes[None, :, 0]
    t = candidate_anchors_cy - gt_boxes[None, :, 1]
    r = gt_boxes[None, :, 2] - candidate_anchors_cx
    b = gt_boxes[None, :, 3] - candidate_anchors_cy
    is_in_gts = torch.stack([l, t, r, b], dim=2).min(dim=2)[0] > 0.01  # [L*topk, M]
    is_pos = is_pos & is_in_gts

    # Assign each anchor to its highest IoU ground-truth if multiple hit
    is_pos_mask = torch.zeros_like(ious, dtype=torch.bool)
    for gt_idx in range(num_gt):
        cand_idx = candidate_idxs[:, gt_idx]
        cand_pos = is_pos[:, gt_idx]
        is_pos_mask[cand_idx[cand_pos], gt_idx] = True

    ious_masked = ious.clone()
    ious_masked[~is_pos_mask] = -1e8

    anchors_to_gt_values, anchors_to_gt_indexs = ious_masked.max(dim=1)

    # Class mapping (-1 used as Background/Negatives)
    cls_labels_per_im = gt_labels[anchors_to_gt_indexs].clone()
    cls_labels_per_im[anchors_to_gt_values == -1e8] = -1
    matched_gts = gt_boxes[anchors_to_gt_indexs]

    return cls_labels_per_im, matched_gts


def compute_centerness_targets(gts, anchors):
    anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
    anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
    l = anchors_cx - gts[:, 0]
    t = anchors_cy - gts[:, 1]
    r = gts[:, 2] - anchors_cx
    b = gts[:, 3] - anchors_cy
    left_right = torch.stack([l, r], dim=1)
    top_bottom = torch.stack([t, b], dim=1)
    centerness = torch.sqrt(
        (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
        * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    )
    return centerness


# -----------------------------------------------------------------------------
# AiTLAS Wrapper
# -----------------------------------------------------------------------------


class ATSS(BaseObjectDetection):
    """
    Implementation of ATSS: Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection
    Based on: https://github.com/sfzhang15/ATSS
    Original paper: https://ieeexplore.ieee.org/document/9156746
    DOI: 10.1109/CVPR42600.2020.00978
    """

    def __init__(self, config):
        super().__init__(config)

        # Config properties
        in_channels = getattr(self.config, "in_channels", 3)
        pretrained = self.config.pretrained
        self.num_classes = self.config.num_classes
        backbone_name = getattr(self.config, "backbone", "resnet50")
        local_model_path = getattr(self.config, "local_model_path", None)
        model_url = "https://drive.google.com/file/d/1t8RLdQ6fsFXa0kzPIQ7541uZeQeMXP73/view"  # getattr(self.config, "model_url", None)

        # 1. Determine backbone initialization strategy
        # We only download ImageNet weights if pretrained=True AND no custom ATSS weights are provided
        load_backbone_pretrained = pretrained and not (model_url or local_model_path)

        self.model = ATSSModel(
            num_classes=self.num_classes,
            pretrained=load_backbone_pretrained,
            backbone_name=backbone_name,
            in_channels=in_channels,
        )
        self.model.to(self.device)

        # 2. Handle full weight loading if pretrained is True
        if pretrained:
            if model_url:
                hub_dir = torch.hub.get_dir()
                checkpoints_dir = os.path.join(hub_dir, "checkpoints")
                os.makedirs(checkpoints_dir, exist_ok=True)

                filename = "ATSS_R_50_FPN_1x.pth"
                cached_file = os.path.join(checkpoints_dir, filename)

                if "drive.google.com" in model_url:
                    parts = model_url.split("/")
                    file_id = parts[parts.index("d") + 1]

                    if not os.path.exists(cached_file):
                        print(f"Downloading weights to hub cache: {cached_file}")
                        self.download_gdrive_to_file(file_id, cached_file)

                print(f"Loading weights from {cached_file}...")
                self.load_atss_weights(cached_file)

    def download_gdrive_to_file(self, file_id, destination):
        """Bypasses Google Drive virus scan warning using a multi-step approach."""
        url = "https://docs.google.com/uc?export=download"

        # 1. Spoof a real browser. Google Drive often blocks default python-requests headers.
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        session = requests.Session()
        session.headers.update(headers)

        # Step 1: Initial request
        response = session.get(url, params={"id": file_id}, stream=True)

        token = None
        # Method A: Check Cookies
        for key, value in session.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        # Method B: Regex on HTML
        if not token:
            # Check for standard confirm token
            match = re.search(r"confirm=([0-9A-Za-z_-]+)", response.text)
            if match:
                token = match.group(1)
            else:
                # Sometimes Google tucks it inside a <form action="...">
                match = re.search(r'action="([^"]+)"', response.text)
                if match and "confirm=" in match.group(1):
                    redirect_url = match.group(1)
                    if redirect_url.startswith("/"):
                        redirect_url = "https://docs.google.com" + redirect_url
                    response = session.get(redirect_url, stream=True)
                    token = "already_handled"

        # Step 2: Second request with the token (if we found one)
        if token and token != "already_handled":
            response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

        # Step 3: The modern User-Content Fallback
        # If we are STILL getting HTML, try Google's newer direct-download format
        if "text/html" in response.headers.get("Content-Type", ""):
            fallback_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            response = session.get(fallback_url, stream=True)

        # Final Verification
        if "text/html" in response.headers.get("Content-Type", ""):
            # Dump the HTML so we aren't flying blind
            debug_file = "debug_gdrive_response.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            raise RuntimeError(
                f"Still returning HTML (Status {response.status_code}). Google may be serving a CAPTCHA or blocking the IP. Check '{debug_file}' to see the exact page."
            )

        # Step 4: Write binary data to the destination
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        print(f"Successfully downloaded weights: {os.path.getsize(destination) / 1024**2:.2f} MB")

    def load_atss_weights(self, checkpoint_path):
        """
        Loads pretrained weights from the original ATSS repository (maskrcnn-benchmark format).
        Includes a robust mapping for ResNet-50 FPN architecture.
        """
        if not os.path.exists(checkpoint_path):
            # If it's a dict (already loaded via load_state_dict_from_url)
            state_dict = (
                checkpoint_path
                if isinstance(checkpoint_path, dict)
                else torch.load(checkpoint_path, map_location=self.device)
            )
        else:
            state_dict = torch.load(checkpoint_path, map_location=self.device)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        new_state_dict = {}
        in_channels = getattr(self.config, "in_channels", 3)

        for k, v in state_dict.items():
            name = k.replace("module.", "")  # Remove DDP prefix

            # 1. Map Backbone (ResNet)
            if "backbone.body" in name:
                # Original: backbone.body.stem.conv1 -> Current: model.backbone.body.conv1
                name = name.replace("backbone.body.stem.", "backbone.body.")
                # Original: backbone.body.layerX -> Current: model.backbone.body.layerX
                name = name.replace("backbone.body.layer", "backbone.body.layer")

            # 2. Map FPN
            if "backbone.fpn" in name:
                # Original fpn_inner2, 3, 4 -> torchvision inner_blocks.0, 1, 2
                name = name.replace("backbone.fpn.fpn_inner2", "backbone.fpn.inner_blocks.0")
                name = name.replace("backbone.fpn.fpn_inner3", "backbone.fpn.inner_blocks.1")
                name = name.replace("backbone.fpn.fpn_inner4", "backbone.fpn.inner_blocks.2")
                # Original fpn_layer2, 3, 4 -> torchvision layer_blocks.0, 1, 2
                name = name.replace("backbone.fpn.fpn_layer2", "backbone.fpn.layer_blocks.0")
                name = name.replace("backbone.fpn.fpn_layer3", "backbone.fpn.layer_blocks.1")
                name = name.replace("backbone.fpn.fpn_layer4", "backbone.fpn.layer_blocks.2")

            # 3. Map P6/P7 (Extra levels)
            if "backbone.fpn.top_blocks.p6" in name:
                name = name.replace("backbone.fpn.top_blocks.p6", "backbone.p6")
            if "backbone.fpn.top_blocks.p7" in name:
                name = name.replace("backbone.fpn.top_blocks.p7", "backbone.p7")

            # 4. Map Head
            if "rpn.head" in name:
                name = name.replace("rpn.head.", "head.")

            # Filter: Skip first conv if in_channels != 3
            if in_channels != 3 and "backbone.body.conv1.weight" in name:
                print(f"Skipping {name} due to input channel mismatch ({in_channels} vs 3)")
                continue

            # Filter: Skip class-specific weights if num_classes != 80 (COCO)
            if self.num_classes != 80 and (
                "cls_logits.weight" in name or "cls_logits.bias" in name
            ):
                print(f"Skipping {name} due to class count mismatch ({self.num_classes} vs 80)")
                continue

            new_state_dict[name] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained ATSS weights. Missing keys: {len(msg.missing_keys)}")
        if len(msg.missing_keys) < 20 and len(msg.missing_keys) > 0:
            print(f"Missing keys: {msg.missing_keys}")

    def forward(self, inputs, targets=None):
        if isinstance(inputs, list):
            target_sizes = torch.stack(
                [torch.tensor([img.shape[-2], img.shape[-1]], device=self.device) for img in inputs]
            )
            samples = nested_tensor_from_tensor_list(inputs).to(self.device)
        else:
            target_sizes = torch.stack(
                [torch.tensor([inputs.shape[-2], inputs.shape[-1]], device=self.device)]
                * inputs.shape[0]
            )
            samples = nested_tensor_from_tensor_list(list(inputs)).to(self.device)

        if self.model.training and targets is not None:
            logits, bbox_reg, centerness, anchors_per_level = self.model(samples.tensors)

            B = samples.tensors.shape[0]
            logits_concat = torch.cat(
                [l.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes) for l in logits],
                dim=1,
            )
            bbox_reg_concat = torch.cat(
                [b.permute(0, 2, 3, 1).reshape(B, -1, 4) for b in bbox_reg], dim=1
            )
            centerness_concat = torch.cat(
                [c.permute(0, 2, 3, 1).reshape(B, -1) for c in centerness], dim=1
            )

            anchors_concat = torch.cat(anchors_per_level, dim=0)
            num_anchors_per_level = [len(a) for a in anchors_per_level]

            targets_one_hot_batch = torch.zeros_like(logits_concat)  # [B, N, C]
            pos_inds_batch = []
            target_boxes_batch = []

            for i in range(B):
                gt_boxes = targets[i]["boxes"].to(self.device)
                gt_labels = targets[i]["labels"].to(self.device)

                target_labels, target_boxes = compute_atss_targets(
                    anchors_concat,
                    gt_boxes,
                    gt_labels,
                    num_anchors_per_level,
                    topk=self.model.topk,
                )

                pos_inds = target_labels >= 0
                targets_one_hot_batch[i, pos_inds, target_labels[pos_inds]] = 1

                pos_inds_batch.append(pos_inds)
                target_boxes_batch.append(target_boxes[pos_inds])

            num_pos_total = max(sum(p.sum().item() for p in pos_inds_batch), 1.0)
            cls_loss = sigmoid_focal_loss(logits_concat, targets_one_hot_batch, num_pos_total)

            reg_loss = 0.0
            centerness_loss = 0.0
            sum_centerness_targets = 0.0
            centerness_targets_all = []

            for i in range(B):
                pos_inds = pos_inds_batch[i]
                if pos_inds.sum() > 0:
                    cent_t = compute_centerness_targets(
                        target_boxes_batch[i], anchors_concat[pos_inds]
                    )
                    sum_centerness_targets += cent_t.sum().item()
                    centerness_targets_all.append(cent_t)
                else:
                    centerness_targets_all.append(torch.zeros((0,), device=self.device))

            sum_centerness_targets = max(sum_centerness_targets, 1.0)

            for i in range(B):
                pos_inds = pos_inds_batch[i]
                if pos_inds.sum() > 0:
                    decoded_preds = self.model.box_coder.decode(
                        bbox_reg_concat[i][pos_inds], anchors_concat[pos_inds]
                    )
                    r_loss = giou_loss(
                        decoded_preds,
                        target_boxes_batch[i],
                        weight=centerness_targets_all[i],
                    )
                    reg_loss += r_loss

                    c_loss = F.binary_cross_entropy_with_logits(
                        centerness_concat[i][pos_inds],
                        centerness_targets_all[i],
                        reduction="sum",
                    )
                    centerness_loss += c_loss

            reg_loss = reg_loss / sum_centerness_targets * 2.0  # ATSS Default REG_LOSS_WEIGHT = 2.0
            centerness_loss = centerness_loss / num_pos_total

            return {
                "classification_loss": cls_loss,
                "regression_loss": reg_loss + centerness_loss,
            }

        else:
            # Inference mode
            logits, bbox_reg, centerness, anchors_per_level = self.model(samples.tensors)

            final_outputs = []
            B = samples.tensors.shape[0]
            for i in range(B):
                boxes_all = []
                scores_all = []
                labels_all = []

                for level in range(len(logits)):
                    l_logits = logits[level][i].permute(1, 2, 0).reshape(-1, self.num_classes)
                    l_bbox_reg = bbox_reg[level][i].permute(1, 2, 0).reshape(-1, 4)
                    l_centerness = centerness[level][i].permute(1, 2, 0).reshape(-1).sigmoid()
                    l_anchors = anchors_per_level[level]

                    l_scores = l_logits.sigmoid() * l_centerness[:, None]

                    # Pre-NMS Threshold Filter
                    candidate_inds = l_scores > 0.05
                    if candidate_inds.sum() == 0:
                        continue

                    scores_flat = l_scores[candidate_inds]
                    candidate_nonzeros = candidate_inds.nonzero(as_tuple=False)
                    box_loc = candidate_nonzeros[:, 0]
                    box_class = candidate_nonzeros[:, 1]

                    # Top-K Filter
                    if len(scores_flat) > 1000:
                        scores_flat, topk_inds = scores_flat.topk(1000, sorted=False)
                        box_loc = box_loc[topk_inds]
                        box_class = box_class[topk_inds]

                    l_boxes_pred = self.model.box_coder.decode(
                        l_bbox_reg[box_loc], l_anchors[box_loc]
                    )

                    boxes_all.append(l_boxes_pred)
                    # Use geometric mean analog format from original
                    scores_all.append(torch.sqrt(scores_flat))
                    labels_all.append(box_class)

                if len(boxes_all) == 0:
                    final_outputs.append(
                        {
                            "boxes": torch.zeros((0, 4), device=self.device),
                            "scores": torch.zeros((0,), device=self.device),
                            "labels": torch.zeros((0,), dtype=torch.long, device=self.device),
                        }
                    )
                    continue

                boxes_all = torch.cat(boxes_all, dim=0)
                scores_all = torch.cat(scores_all, dim=0)
                labels_all = torch.cat(labels_all, dim=0)

                # Clip boxes to image boundaries
                h, w = target_sizes[i]
                boxes_all[:, 0::2] = boxes_all[:, 0::2].clamp(min=0, max=w)
                boxes_all[:, 1::2] = boxes_all[:, 1::2].clamp(min=0, max=h)

                # Exclude malformed boxes
                ws = boxes_all[:, 2] - boxes_all[:, 0]
                hs = boxes_all[:, 3] - boxes_all[:, 1]
                keep_size = (ws > 0) & (hs > 0)
                boxes_all = boxes_all[keep_size]
                scores_all = scores_all[keep_size]
                labels_all = labels_all[keep_size]

                # Batched NMS (Post-NMS Filter)
                keep = torchvision.ops.batched_nms(boxes_all, scores_all, labels_all, 0.6)
                keep = keep[:100]

                final_outputs.append(
                    {
                        "boxes": boxes_all[keep],
                        "scores": scores_all[keep],
                        "labels": labels_all[keep],
                    }
                )

            return final_outputs
