"""Sparse R-CNN: End-to-End Object Detection with Learnable Proposals"""

import copy
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.boxes import box_area

from ..base.object_detection import BaseObjectDetection


# -----------------------------------------------------------------------------
# Utils & Helper Functions (from projects/SparseRCNN/sparsercnn/util/misc.py and
# projects/SparseRCNN/sparsercnn/util/box_ops.py)
# -----------------------------------------------------------------------------


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    # Ensure w and h are at least a small epsilon to avoid degenerate boxes
    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    xyxy = torch.stack(b, dim=-1)
    # Ensure x2 >= x1 and y2 >= y1 by clamping
    x1, y1, x2, y2 = xyxy.unbind(-1)
    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    """
    # Ensure boxes are valid by clamping (creates new tensors, no in-place ops)
    x1_1, y1_1, x2_1, y2_1 = boxes1.unbind(-1)
    x1_1 = torch.min(x1_1, x2_1)
    y1_1 = torch.min(y1_1, y2_1)
    x2_1 = torch.max(x1_1, x2_1)
    y2_1 = torch.max(y1_1, y2_1)
    boxes1 = torch.stack([x1_1, y1_1, x2_1, y2_1], dim=-1)

    x1_2, y1_2, x2_2, y2_2 = boxes2.unbind(-1)
    x1_2 = torch.min(x1_2, x2_2)
    y1_2 = torch.min(y1_2, y2_2)
    x2_2 = torch.max(x1_2, x2_2)
    y2_2 = torch.max(y1_2, y2_2)
    boxes2 = torch.stack([x1_2, y1_2, x2_2, y2_2], dim=-1)

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


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


# -----------------------------------------------------------------------------
# Dynamic Head & Dynamic Conv Modules (from projects/SparseRCNN/sparsercnn/head.py)
# -----------------------------------------------------------------------------


class DynamicConv(nn.Module):
    def __init__(self, hidden_dim, dim_dynamic, num_dynamic, pooler_resolution):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.hidden_dim * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        """
        pro_features: (1,  N * nr_boxes, self.hidden_dim)
        roi_features: (pooler_resolution**2, N * nr_boxes, self.hidden_dim)
        """
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, : self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params :].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


class RCNNHead(nn.Module):
    def __init__(
        self,
        d_model,
        num_classes,
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        activation="relu",
        num_cls=1,
        num_reg=3,
        dim_dynamic=64,
        num_dynamic=2,
        pooler_resolution=7,
        scale_clamp=math.log(100000.0 / 16),
    ):
        super().__init__()
        self.d_model = d_model

        # Dynamic
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(d_model, dim_dynamic, num_dynamic, pooler_resolution)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

        # Classification Module
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # Regression Module
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # Prediction
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = (2.0, 2.0, 1.0, 1.0)

    def forward(self, features, bboxes, pro_features, pooler, image_shapes):
        N, nr_boxes = bboxes.shape[:2]

        # roi_feature (Standard Torchvision RoI Pooler Interface)
        proposal_boxes = list(bboxes)
        roi_features = pooler(features, proposal_boxes, image_shapes)
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact
        pro_features = (
            pro_features.view(nr_boxes, N, self.d_model)
            .permute(1, 0, 2)
            .reshape(1, N * nr_boxes, self.d_model)
        )
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()

        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (
            class_logits.view(N, nr_boxes, -1),
            pred_bboxes.view(N, nr_boxes, -1),
            obj_features,
        )

    def apply_deltas(self, deltas, boxes):
        boxes = boxes.to(deltas.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent large exponential predictions
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DynamicHead(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim,
        dim_feedforward,
        nhead,
        dropout,
        activation,
        num_heads,
        num_cls,
        num_reg,
        dim_dynamic,
        num_dynamic,
        pooler_resolution,
        return_intermediate,
    ):
        super().__init__()
        # Standard torchvision Pooler matching Detectron2's interface
        self.box_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=pooler_resolution,
            sampling_ratio=2,
        )

        rcnn_head = RCNNHead(
            d_model=hidden_dim,
            num_classes=num_classes,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            num_cls=num_cls,
            num_reg=num_reg,
            dim_dynamic=dim_dynamic,
            num_dynamic=num_dynamic,
            pooler_resolution=pooler_resolution,
        )

        self.head_series = _get_clones(rcnn_head, num_heads)
        self.return_intermediate = return_intermediate
        self.num_classes = num_classes

        # Focal Loss bias initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for head in self.head_series:
            nn.init.constant_(head.class_logits.bias, bias_value)

    def forward(self, features, init_bboxes, init_features, image_shapes):
        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(init_bboxes)
        bboxes = init_bboxes

        init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler, image_shapes
            )

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


# -----------------------------------------------------------------------------
# Matcher & Loss (from projects/SparseRCNN/sparsercnn/loss.py)
# -----------------------------------------------------------------------------


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])

        alpha = self.focal_alpha
        gamma = self.focal_gamma
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Replace any NaN or Inf values with large finite numbers
        cost_class = torch.nan_to_num(cost_class, nan=1.0, posinf=1.0, neginf=-1.0)
        cost_bbox = torch.nan_to_num(cost_bbox, nan=1.0, posinf=1.0, neginf=0.0)
        cost_giou = torch.nan_to_num(cost_giou, nan=1.0, posinf=1.0, neginf=-1.0)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1

        class_loss = sigmoid_focal_loss(
            src_logits,
            labels,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        return {"loss_ce": class_loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes_xyxy"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size

        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {"labels": self.loss_labels, "boxes": self.loss_boxes}
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# -----------------------------------------------------------------------------
# Main Sparse R-CNN Core Architect (from projects/SparseRCNN/sparsercnn/detector.py)
# -----------------------------------------------------------------------------


class SparseRCNNModel(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes,
        num_proposals,
        hidden_dim,
        dim_feedforward,
        nhead,
        dropout,
        activation,
        num_heads,
        num_cls,
        num_reg,
        dim_dynamic,
        num_dynamic,
        pooler_resolution,
        deep_supervision,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_proposals = num_proposals
        self.hidden_dim = hidden_dim
        self.deep_supervision = deep_supervision

        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        self.head = DynamicHead(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads,
            num_cls=num_cls,
            num_reg=num_reg,
            dim_dynamic=dim_dynamic,
            num_dynamic=num_dynamic,
            pooler_resolution=pooler_resolution,
            return_intermediate=deep_supervision,
        )

    def forward(self, tensor):
        # We calculate the structural variables dynamically inside this wrapper instead of external param
        b, c, h, w = tensor.shape
        images_whwh = (
            torch.tensor([w, h, w, h], dtype=torch.float32, device=tensor.device)
            .unsqueeze(0)
            .repeat(b, 1)
        )
        image_shapes = [(h, w)] * b

        # 1. Feature Extraction
        src = self.backbone(tensor)
        # Using feature maps from the backbone resnet_fpn configuration
        features = {k: src[k] for k in ["0", "1", "2", "3"]}

        # 2. Proposals scaling
        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        # 3. Predict Dynamic Head
        outputs_class, outputs_coord = self.head(
            features, proposal_boxes, self.init_proposal_features.weight, image_shapes
        )
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        if self.deep_supervision:
            output["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        return output


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------


class SparseRCNN(BaseObjectDetection):
    """
    Implementation of Sparse R-CNN: End-to-End Object Detection with Learnable Proposals
    Based on: https://github.com/PeizeSun/SparseR-CNN
    Original paper: https://ieeexplore.ieee.org/document/9577670
    DOI: 10.1109/CVPR46437.2021.01422
    """

    def __init__(self, config):
        super().__init__(config)

        # Config properties (Standard SparseRCNN Defaults)
        self.num_classes = self.config.num_classes
        self.num_proposals = getattr(self.config, "num_proposals", 300)
        self.hidden_dim = getattr(self.config, "hidden_dim", 256)
        self.num_heads = getattr(self.config, "num_heads", 6)

        backbone_name = getattr(self.config, "backbone", "resnet50")
        pretrained = self.config.pretrained
        model_url = (
            "https://github.com/PeizeSun/SparseR-CNN/releases/download/v0.1/r50_300pro_3x_model.pth"
        )

        # Determine backbone initialization strategy
        if pretrained and not model_url:
            backbone_weights = "DEFAULT"
        else:
            backbone_weights = None

        # 1. Build Backbone (using Torchvision standard FPN identical structurally)
        backbone = resnet_fpn_backbone(backbone_name=backbone_name, weights=backbone_weights)

        # 2. Initialize Main Sparse R-CNN Component
        self.model = SparseRCNNModel(
            backbone=backbone,
            num_classes=self.num_classes,
            num_proposals=self.num_proposals,
            hidden_dim=self.hidden_dim,
            dim_feedforward=getattr(self.config, "dim_feedforward", 2048),
            nhead=getattr(self.config, "nheads", 8),
            dropout=getattr(self.config, "dropout", 0.1),
            activation=getattr(self.config, "activation", "relu"),
            num_heads=self.num_heads,
            num_cls=getattr(self.config, "num_cls", 1),
            num_reg=getattr(self.config, "num_reg", 3),
            dim_dynamic=getattr(self.config, "dim_dynamic", 64),
            num_dynamic=getattr(self.config, "num_dynamic", 2),
            pooler_resolution=getattr(self.config, "pooler_resolution", 7),
            deep_supervision=getattr(self.config, "deep_supervision", True),
        )
        self.model.to(self.device)

        # 3. Build Criterion (Loss logic)
        class_weight = getattr(self.config, "class_weight", 2.0)
        giou_weight = getattr(self.config, "giou_weight", 2.0)
        l1_weight = getattr(self.config, "l1_weight", 5.0)

        matcher = HungarianMatcher(
            cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_bbox": l1_weight,
            "loss_giou": giou_weight,
        }
        if self.model.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.sparsercnn_criterion = SetCriterion(
            num_classes=self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=["labels", "boxes"],
        )
        self.sparsercnn_criterion.to(self.device)

        # 4. Handle full weight loading if pretrained is True
        if pretrained and model_url:
            self.load_sparsercnn_weights(model_url)

    def load_sparsercnn_weights(self, model_url):
        """
        Loads pretrained weights for Sparse R-CNN.
        Includes mapping from Detectron2/Official format to SparseRCNNModel wrapper.
        """
        print(f"Loading weights from {model_url}...")
        state_dict = load_state_dict_from_url(model_url, map_location=self.device)

        if "model" in state_dict:
            state_dict = state_dict["model"]

        new_state_dict = {}
        in_channels = getattr(self.config, "in_channels", 3)

        for k, v in state_dict.items():
            name = k

            # Map Backbone
            if "backbone.bottom_up" in name:
                # Detectron2 Backbone -> Torchvision FPN Backbone
                name = name.replace(
                    "backbone.bottom_up.stem.conv1.weight", "backbone.body.conv1.weight"
                )
                name = name.replace("backbone.bottom_up.stem.conv1.norm", "backbone.body.bn1")

                # Layers mapping: res2, res3, res4, res5 -> layer1, layer2, layer3, layer4
                for i in range(2, 6):
                    name = name.replace(
                        f"backbone.bottom_up.res{i}.", f"backbone.body.layer{i - 1}."
                    )

                # Detectron2 ResNet naming vs Torchvision
                # Conv Norm mapping: conv1.norm -> bn1, conv2.norm -> bn2, conv3.norm -> bn3
                name = name.replace(".conv1.norm.", ".bn1.")
                name = name.replace(".conv2.norm.", ".bn2.")
                name = name.replace(".conv3.norm.", ".bn3.")

                # Shortcut mapping
                name = name.replace(".shortcut.weight", ".downsample.0.weight")
                name = name.replace(".shortcut.norm.", ".downsample.1.")

            # Map FPN
            if "backbone.fpn_" in name:
                # Detectron2 FPN uses fpn_lateral (1x1 conv) and fpn_output (3x3 conv)
                # Torchvision FPN uses inner_blocks (1x1 conv) and layer_blocks (3x3 conv)
                # Note: Torchvision wraps Conv2d in Conv2dNormActivation, so we need .0.0 for weights
                # Mapping: fpn_lateral -> inner_blocks.0.0, fpn_output -> layer_blocks.0.0

                # fpn_lateral2,3,4,5 -> inner_blocks.0,1,2,3.0 (both are 1x1 convs)
                name = name.replace("backbone.fpn_lateral2", "backbone.fpn.inner_blocks.0.0")
                name = name.replace("backbone.fpn_lateral3", "backbone.fpn.inner_blocks.1.0")
                name = name.replace("backbone.fpn_lateral4", "backbone.fpn.inner_blocks.2.0")
                name = name.replace("backbone.fpn_lateral5", "backbone.fpn.inner_blocks.3.0")

                # fpn_output2,3,4,5 -> layer_blocks.0,1,2,3.0 (both are 3x3 convs)
                name = name.replace("backbone.fpn_output2", "backbone.fpn.layer_blocks.0.0")
                name = name.replace("backbone.fpn_output3", "backbone.fpn.layer_blocks.1.0")
                name = name.replace("backbone.fpn_output4", "backbone.fpn.layer_blocks.2.0")
                name = name.replace("backbone.fpn_output5", "backbone.fpn.layer_blocks.3.0")

                # Also handle fpn_inner* aliases (some checkpoints use this naming)
                name = name.replace("backbone.fpn_inner2", "backbone.fpn.inner_blocks.0.0")
                name = name.replace("backbone.fpn_inner3", "backbone.fpn.inner_blocks.1.0")
                name = name.replace("backbone.fpn_inner4", "backbone.fpn.inner_blocks.2.0")
                name = name.replace("backbone.fpn_inner5", "backbone.fpn.inner_blocks.3.0")

            # Map Heads - checkpoint already has "head." prefix, no mapping needed
            # The checkpoint uses: head.head_series.X.* which matches our structure

            # Filter: Skip first conv if in_channels != 3
            if in_channels != 3 and "backbone.body.conv1.weight" in name:
                print(f"Skipping {name} due to input channel mismatch ({in_channels} vs 3)")
                continue

            # Filter: Skip class-specific weights if num_classes != 80 (COCO)
            if self.num_classes != 80 and "class_logits" in name:
                print(f"Skipping {name} due to class count mismatch ({self.num_classes} vs 80)")
                continue

            new_state_dict[name] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained Sparse R-CNN weights. Missing keys: {len(msg.missing_keys)}")
        if len(msg.missing_keys) < 20 and len(msg.missing_keys) > 0:
            print(f"Missing keys: {msg.missing_keys}")

    def forward(self, inputs, targets=None):
        # 1. Standardize Inputs
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

        images_whwh = target_sizes[:, [1, 0, 1, 0]].to(torch.float32)

        # Spatial size fed to backbone is the padded mask
        padded_h, padded_w = samples.tensors.shape[-2:]
        image_shapes = [(padded_h, padded_w)] * samples.tensors.shape[0]

        # 2. Mode Route
        if self.model.training and targets is not None:
            new_targets = []
            for target, size in zip(targets, target_sizes):
                h, w = size
                image_size_xyxy = torch.as_tensor(
                    [w, h, w, h], dtype=torch.float, device=self.device
                )

                gt_boxes_xyxy = target["boxes"].to(self.device)
                gt_classes = target["labels"].to(self.device)

                # Create correctly normalized variants expected by SparseRCNN
                if gt_boxes_xyxy.shape[0] > 0:
                    gt_boxes = box_xyxy_to_cxcywh(gt_boxes_xyxy) / image_size_xyxy
                else:
                    gt_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)

                new_targets.append(
                    {
                        "labels": gt_classes,
                        "boxes": gt_boxes,
                        "boxes_xyxy": gt_boxes_xyxy,
                        "image_size_xyxy": image_size_xyxy,
                        "image_size_xyxy_tgt": image_size_xyxy.unsqueeze(0).repeat(
                            len(gt_boxes_xyxy), 1
                        ),
                    }
                )

            outputs = self.model(samples.tensors)
            loss_dict = self.sparsercnn_criterion(outputs, new_targets)

            # 3. Combine and weight losses
            weight_dict = self.sparsercnn_criterion.weight_dict
            classification_loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if "loss_ce" in k and k in weight_dict
            )
            regression_loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if ("loss_bbox" in k or "loss_giou" in k) and k in weight_dict
            )

            return {
                "classification_loss": classification_loss,
                "regression_loss": regression_loss,
            }

        else:
            # Inference mode
            outputs = self.model(samples.tensors)

            box_cls = outputs["pred_logits"]
            box_pred = outputs["pred_boxes"]

            scores = torch.sigmoid(box_cls)
            labels = (
                torch.arange(self.num_classes, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_proposals, 1)
                .flatten(0, 1)
            )

            final_outputs = []

            # Formulate outputs strictly meeting AiTLAS evaluation routines
            for scores_per_image, box_pred_per_image in zip(scores, box_pred):
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
                    self.num_proposals, sorted=False
                )
                labels_per_image = labels[topk_indices]
                box_pred_per_image = (
                    box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                )
                box_pred_per_image = box_pred_per_image[topk_indices]

                mask = scores_per_image > 0.05

                if not mask.any():
                    final_outputs.append(
                        {
                            "boxes": torch.zeros((0, 4), device=self.device),
                            "scores": torch.zeros((0,), device=self.device),
                            "labels": torch.zeros((0,), device=self.device),
                        }
                    )
                else:
                    final_outputs.append(
                        {
                            "boxes": box_pred_per_image[mask],
                            "scores": scores_per_image[mask],
                            "labels": labels_per_image[mask],
                        }
                    )

            return final_outputs
