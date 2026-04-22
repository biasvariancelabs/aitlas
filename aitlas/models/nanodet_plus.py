"""NanoDet-Plus: Super fast and high accuracy lightweight anchor-free object detection model"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops.boxes import batched_nms, box_iou, nms

from ..base import BaseObjectDetection


# -----------------------------------------------------------------------------
# Constants (hardcoded for nanodet-plus-m_416)
# -----------------------------------------------------------------------------
STRIDES = [8, 16, 32, 64]
REG_MAX = 7
FPN_OUT_CHANNELS = 96
KERNEL_SIZE = 5
BACKBONE_OUT_CHANNELS = [116, 232, 464]  # ShuffleNetV2 1.0x stages 2,3,4
NUM_EXTRA_LEVEL = 1
STACKED_CONVS = 2
AUX_STACKED_CONVS = 4
AUX_FEAT_CHANNELS = 192
TOPK = 13
IOU_FACTOR = 3.0
DETACH_EPOCH = 10


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Calculate distance from points to bounding box boundaries."""
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance predictions to bounding boxes."""
    if points.dim() == 3:
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
    else:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox_overlaps(bboxes1, bboxes2):
    """Calculate IoU between two sets of bounding boxes."""
    return box_iou(bboxes1, bboxes2)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


def quality_focal_loss(pred, target, beta=2.0, reduction="none"):
    """Quality Focal Loss (QFL) for classification."""
    if isinstance(target, tuple):
        labels, scores = target
        assert pred.size(0) == labels.size(0) == scores.size(0)
    else:
        labels = None
        scores = target

    pred_sigmoid = pred.sigmoid()
    zerolabel = pred.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * pred_sigmoid.pow(beta)

    if labels is not None:
        num_classes = pred.shape[1]
        pos = (labels >= 0) & (labels < num_classes)

        if pos.any():
            pos_labels = labels[pos]
            pos_pred = pred[pos]
            pos_scores = scores[pos]
            target_onehot = pred.new_zeros(pos_pred.shape)
            target_onehot.scatter_(1, pos_labels.unsqueeze(1).long(), 1.0)

            loss[pos] = F.binary_cross_entropy_with_logits(
                pos_pred, target_onehot, reduction="none"
            ) * (pos_scores.unsqueeze(1) - pred_sigmoid[pos]).pow(beta)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss


def distribution_focal_loss(
    pred, label, weight=None, reduction="none", avg_factor=None
):
    """Distribution Focal Loss (DFL) for regression."""
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()

    loss = (
        F.cross_entropy(pred, dis_left, reduction="none") * weight_left
        + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )

    if weight is not None:
        loss = loss * weight
    if avg_factor is not None and avg_factor > 0:
        loss = loss / avg_factor
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def giou_loss(pred, target, weight=None, reduction="none", avg_factor=None, eps=1e-7):
    """Generalized Intersection over Union Loss."""
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosed_lt = torch.min(pred[:, :2], target[:, :2])
    enclosed_rb = torch.max(pred[:, 2:], target[:, 2:])
    enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)

    intersect = wh[:, 0] * wh[:, 1]
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_target - intersect
    iou = intersect / (union + eps)
    giou = iou - (enclosed_wh[:, 0] * enclosed_wh[:, 1] - union) / (
        enclosed_wh[:, 0] * enclosed_wh[:, 1] + eps
    )
    loss = 1 - giou

    if weight is not None:
        loss = loss * weight
    if avg_factor is not None and avg_factor > 0:
        loss = loss / avg_factor
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


# -----------------------------------------------------------------------------
# Assigner
# -----------------------------------------------------------------------------


class DynamicSoftLabelAssigner(nn.Module):
    """Dynamic Soft Label Assigner from NanoDet-Plus."""

    def __init__(self, topk=TOPK, iou_factor=IOU_FACTOR):
        super().__init__()
        self.topk = topk
        self.iou_factor = iou_factor

    @torch.no_grad()
    def forward(
        self,
        pred_scores,
        priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        """Assign gt to priors with dynamic soft label assignment."""
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.size(0), decoded_bboxes.size(0)

        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        max_overlaps = decoded_bboxes.new_full((num_bboxes,), -INF, dtype=torch.float32)

        if num_gt == 0 or num_bboxes == 0:
            return (
                assigned_gt_inds,
                assigned_labels,
                max_overlaps,
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                0,
            )

        # Check if prior center is inside gt
        prior_center = priors[:, :2]
        deltas = torch.cat(
            [
                prior_center[:, None] - gt_bboxes[:, :2],
                gt_bboxes[:, 2:] - prior_center[:, None],
            ],
            dim=-1,
        )
        is_in_gts = deltas.min(dim=-1).values > 0
        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_valid == 0:
            return (
                assigned_gt_inds,
                assigned_labels,
                max_overlaps,
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                0,
            )

        # Compute cost matrix
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)
        cost_matrix = cls_cost.sum(dim=-1) + iou_cost * self.iou_factor

        # Dynamic k matching
        matching_matrix = torch.zeros_like(cost_matrix)
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost_matrix[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        # Resolve multiple GTs mapped to same anchor
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost_matrix[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] = 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0

        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]

        pos_inds = torch.where(valid_mask)[0]
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps[valid_mask] = matched_pred_ious
        neg_inds = torch.where((assigned_gt_inds == 0) & valid_mask)[0]

        return (
            assigned_gt_inds,
            assigned_labels,
            max_overlaps,
            pos_inds,
            neg_inds,
            pos_inds.size(0),
        )


# -----------------------------------------------------------------------------
# Integral Layer for DFL
# -----------------------------------------------------------------------------


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution."""

    def __init__(self, reg_max=REG_MAX):
        super().__init__()
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result."""
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, -1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x


# -----------------------------------------------------------------------------
# Model Modules
# -----------------------------------------------------------------------------


class InvertedResidualLeaky(nn.Module):
    """InvertedResidual with LeakyReLU activation."""

    def __init__(self, inp, oup, stride, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self._depthwise_conv(inp, inp, 3, stride, 1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                act,
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if stride > 1 else branch_features, branch_features, 1, bias=False
            ),
            nn.BatchNorm2d(branch_features),
            act,
            self._depthwise_conv(branch_features, branch_features, 3, stride, 1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, bias=False),
            nn.BatchNorm2d(branch_features),
            act,
        )

    @staticmethod
    def _depthwise_conv(i, o, k, s=1, p=0):
        return nn.Conv2d(i, o, k, s, p, groups=i, bias=False)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        # Channel shuffle
        b, c, h, w = out.size()
        out = (
            out.view(b, 2, c // 2, h, w).transpose(1, 2).contiguous().view(b, -1, h, w)
        )
        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 Backbone with LeakyReLU and multi-stage output."""

    def __init__(self, model_size="1.0x", out_stages=(2, 3, 4), pretrain=True):
        super().__init__()
        assert set(out_stages).issubset((2, 3, 4))
        self.out_stages = out_stages

        stages_out_channels = {
            "0.5x": [24, 48, 96, 192, 1024],
            "1.0x": [24, 116, 232, 464, 1024],
            "1.5x": [24, 176, 352, 704, 1024],
            "2.0x": [24, 244, 488, 976, 2048],
        }.get(model_size)

        if stages_out_channels is None:
            raise NotImplementedError(f"model_size {model_size} not implemented")

        act = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, stages_out_channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(stages_out_channels[0]),
            act,
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        stage_names = ["stage2", "stage3", "stage4"]
        input_channels = stages_out_channels[0]
        for name, repeats, out_ch in zip(
            stage_names, [4, 8, 4], stages_out_channels[1:]
        ):
            seq = [InvertedResidualLeaky(input_channels, out_ch, 2, act)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidualLeaky(out_ch, out_ch, 1, act))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = out_ch

        if pretrain:
            self._load_pretrained(model_size)

    def _load_pretrained(self, model_size):
        try:
            weights = {
                "0.5x": models.ShuffleNet_V2_Weights.SHUFFLENET_V2_X0_5,
                "1.0x": models.ShuffleNet_V2_Weights.SHUFFLENET_V2_X1,
                "1.5x": models.ShuffleNet_V2_Weights.SHUFFLENET_V2_X1_5,
                "2.0x": models.ShuffleNet_V2_Weights.SHUFFLENET_V2_X2_0,
            }.get(model_size)
            if weights:
                pretrained_state = weights.get_state_dict(progress=True)
                pretrained_state = {
                    k: v
                    for k, v in pretrained_state.items()
                    if not k.startswith("fc") and not k.startswith("conv5")
                }
                self.load_state_dict(pretrained_state, strict=False)
        except Exception:
            print(
                f"Warning: Could not load pretrained weights for ShuffleNetV2 {model_size}"
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            x = getattr(self, f"stage{i}")(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)


class ConvModule(nn.Module):
    """Conv + BN + Activation module."""

    def __init__(
        self,
        in_c,
        out_c,
        k=1,
        s=1,
        p=0,
        groups=1,
        depthwise=False,
        act=nn.LeakyReLU(0.1, inplace=True),
    ):
        super().__init__()
        self.act = act

        if depthwise:
            self.depthwise = nn.Conv2d(in_c, in_c, k, s, p, groups=in_c, bias=False)
            self.dwnorm = nn.BatchNorm2d(in_c)
            self.pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
            self.pwnorm = nn.BatchNorm2d(out_c)
        else:
            self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        if hasattr(self, "depthwise"):
            x = self.depthwise(x)
            x = self.dwnorm(x)
            if self.act is not None:
                x = self.act(x)
            x = self.pointwise(x)
            x = self.pwnorm(x)
            if self.act is not None:
                x = self.act(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            if self.act is not None:
                x = self.act(x)
        return x


class GhostModule(nn.Module):
    """GhostModule for feature generation."""

    def __init__(self, inp, oup, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        init_channels = math.ceil(oup / 2)

        layers_primary = [
            nn.Conv2d(inp, init_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(init_channels),
        ]
        if act is not None:
            layers_primary.append(act)
        self.primary_conv = nn.Sequential(*layers_primary)

        layers_cheap = [
            nn.Conv2d(
                init_channels, init_channels, 3, 1, 1, groups=init_channels, bias=False
            ),
            nn.BatchNorm2d(init_channels),
        ]
        if act is not None:
            layers_cheap.append(act)
        self.cheap_operation = nn.Sequential(*layers_cheap)

        self.oup = oup

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, : self.oup, :, :]


class GhostBottleneck(nn.Module):
    """GhostBottleneck used in GhostPAN."""

    def __init__(
        self, in_chs, mid_chs, out_chs, stride=1, act=nn.LeakyReLU(0.1, inplace=True)
    ):
        super().__init__()
        self.ghost1 = GhostModule(in_chs, mid_chs, act)

        self.conv_dw = (
            nn.Sequential(
                nn.Conv2d(mid_chs, mid_chs, 5, stride, 2, groups=mid_chs, bias=False),
                nn.BatchNorm2d(mid_chs),
            )
            if stride > 1
            else None
        )

        self.ghost2 = GhostModule(mid_chs, out_chs, act=None)

        if in_chs != out_chs or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, 5, stride, 2, groups=in_chs, bias=False),  # 0
                nn.BatchNorm2d(in_chs),  # 1
                nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False),  # 2
                nn.BatchNorm2d(out_chs),  # 3
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.conv_dw:
            x = self.conv_dw(x)
        x = self.ghost2(x)
        shortcut_out = self.shortcut(residual)
        return x + shortcut_out


class GhostPAN(nn.Module):
    """Path Aggregation Network with Ghost block."""

    def __init__(
        self,
        in_channels,
        out_channels=FPN_OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        num_extra_level=NUM_EXTRA_LEVEL,
        act=nn.LeakyReLU(0.1, inplace=True),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Reduce layers
        self.reduce_layers = nn.ModuleList(
            [ConvModule(ch, out_channels, 1, act=act) for ch in in_channels]
        )

        # Top-down path
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.top_down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ModuleList(
                        [
                            GhostBottleneck(
                                out_channels * 2, out_channels, out_channels, act=act
                            )
                        ]
                    )
                )
                for _ in range(len(in_channels) - 1)
            ]
        )

        # Bottom-up path
        self.downsamples = nn.ModuleList(
            [
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size,
                    2,
                    kernel_size // 2,
                    depthwise=True,
                    act=act,
                )
                for _ in range(len(in_channels) - 1)
            ]
        )
        self.bottom_up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ModuleList(
                        [
                            GhostBottleneck(
                                out_channels * 2, out_channels, out_channels, act=act
                            )
                        ]
                    )
                )
                for _ in range(len(in_channels) - 1)
            ]
        )

        # Extra layers (paired convs: one for input, one for output, then add)
        self.extra_in_convs = nn.ModuleList(
            [
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size,
                    2,
                    kernel_size // 2,
                    depthwise=True,
                    act=act,
                )
                for _ in range(num_extra_level)
            ]
        )
        self.extra_out_convs = nn.ModuleList(
            [
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size,
                    2,
                    kernel_size // 2,
                    depthwise=True,
                    act=act,
                )
                for _ in range(num_extra_level)
            ]
        )

    def forward(self, inputs):
        # Reduce channels
        inputs = [reduce(x) for x, reduce in zip(inputs, self.reduce_layers)]

        # Top-down
        inner_outs = [inputs[-1]]
        for i, block in enumerate(self.top_down_blocks):
            idx = len(self.in_channels) - 2 - i
            inner_out = block[0][0](
                torch.cat([self.upsample(inner_outs[0]), inputs[idx]], 1)
            )
            inner_outs.insert(0, inner_out)

        # Bottom-up
        outs = [inner_outs[0]]
        for i, (down, block) in enumerate(zip(self.downsamples, self.bottom_up_blocks)):
            out = block[0][0](torch.cat([down(outs[-1]), inner_outs[i + 1]], 1))
            outs.append(out)

        # Extra layers
        for extra_in, extra_out in zip(self.extra_in_convs, self.extra_out_convs):
            outs.append(extra_in(inputs[-1]) + extra_out(outs[-1]))

        return tuple(outs)


# -----------------------------------------------------------------------------
# Heads
# -----------------------------------------------------------------------------


class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus."""

    def __init__(
        self,
        num_classes,
        input_channel=FPN_OUT_CHANNELS,
        feat_channels=FPN_OUT_CHANNELS,
        stacked_convs=STACKED_CONVS,
        kernel_size=KERNEL_SIZE,
        strides=STRIDES,
        reg_max=REG_MAX,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max

        self.distribution_project = Integral(reg_max)
        self.assigner = DynamicSoftLabelAssigner()

        # Build convs and heads for each stride
        self.cls_convs = nn.ModuleList()
        self.gfl_cls = nn.ModuleList()

        act = nn.LeakyReLU(0.1, inplace=True)
        for _ in strides:
            cls_convs = nn.ModuleList(
                [
                    ConvModule(
                        self.in_channels if i == 0 else self.feat_channels,
                        self.feat_channels,
                        kernel_size,
                        1,
                        kernel_size // 2,
                        depthwise=True,
                        act=act,
                    )
                    for i in range(stacked_convs)
                ]
            )
            self.cls_convs.append(cls_convs)
            self.gfl_cls.append(
                nn.Conv2d(self.feat_channels, num_classes + 4 * (reg_max + 1), 1)
            )

        self.init_weights()

    def init_weights(self):
        for gfl_cls in self.gfl_cls:
            nn.init.normal_(gfl_cls.weight, std=0.01)
            if gfl_cls.bias is not None:
                nn.init.constant_(gfl_cls.bias, -4.595)

    def forward(self, feats):
        outputs = []
        for feat, cls_convs, gfl_cls in zip(feats, self.cls_convs, self.gfl_cls):
            for conv in cls_convs:
                feat = conv(feat)
            outputs.append(gfl_cls(feat).flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses."""
        device = preds.device
        batch_size = preds.shape[0]
        gt_bboxes = [t["boxes"] for t in gt_meta]
        gt_labels = [t["labels"] for t in gt_meta]

        input_height, input_width = (
            gt_meta[0]["img"].shape[2:] if "img" in gt_meta[0] else (416, 416)
        )
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                (math.ceil(input_height / s), math.ceil(input_width / s)),
                s,
                dtype=torch.float32,
                device=device,
            )
            for i, s in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            batch_assign_res = self.target_assign(
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )
        else:
            batch_assign_res = self.target_assign(
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )

        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, batch_assign_res
        )

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v

        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        """Compute loss from assignment results."""
        labels, label_weights, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(num_pos, 1.0)

        labels = torch.cat(labels, dim=0)
        label_weights = torch.cat(label_weights, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        dist_targets = torch.cat(dist_targets, dim=0)

        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = (
            quality_focal_loss(cls_preds, (labels, label_weights), reduction="sum")
            / num_total_samples
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)
        loss_bbox, loss_dfl = reg_preds.sum() * 0, reg_preds.sum() * 0

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(weight_targets.sum().item(), 1.0)
            loss_bbox = giou_loss(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                reduction="sum",
                avg_factor=bbox_avg_factor,
            )
            loss_dfl = distribution_focal_loss(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                reduction="sum",
                avg_factor=4.0 * bbox_avg_factor,
            )

        return loss_qfl + loss_bbox + loss_dfl, dict(
            loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl
        )

    def target_assign(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """Assign targets for a batch of images."""
        (
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_dist_targets,
            num_pos_total,
        ) = ([], [], [], [], 0)

        for i in range(cls_preds.shape[0]):
            labels, label_weights, bbox_targets, dist_targets, num_pos = (
                self.target_assign_single_img(
                    cls_preds[i],
                    center_priors[i],
                    decoded_bboxes[i],
                    gt_bboxes[i],
                    gt_labels[i],
                )
            )
            all_labels.append(labels)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_dist_targets.append(dist_targets)
            num_pos_total += num_pos

        return (
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_dist_targets,
            num_pos_total,
        )

    def target_assign_single_img(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """Compute targets for one image."""
        if len(gt_bboxes) == 0:
            num_priors = center_priors.size(0)
            return (
                center_priors.new_full(
                    (num_priors,), self.num_classes, dtype=torch.long
                ),
                center_priors.new_zeros(num_priors, dtype=torch.float),
                torch.zeros_like(center_priors),
                torch.zeros_like(center_priors),
                0,
            )

        assigned_gt_inds, assigned_labels, max_overlaps, pos_inds, neg_inds, num_pos = (
            self.assigner(
                cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels
            )
        )

        num_priors = center_priors.size(0)
        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_weights = center_priors.new_zeros(num_priors, dtype=torch.float)

        if len(pos_inds) > 0:
            pos_gt_bboxes = gt_bboxes[assigned_gt_inds[pos_inds] - 1]
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            strides = center_priors[pos_inds, 2:].mean(dim=1, keepdim=True)
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes) / strides
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
            label_weights[pos_inds] = max_overlaps[pos_inds]

        return labels, label_weights, bbox_targets, dist_targets, num_pos

    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate center priors for a single level feature map."""
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        y, x = y.flatten(), x.flatten()
        strides = x.new_full((x.numel(),), stride)
        return (
            torch.stack([x, y, strides, strides], dim=-1)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )


class SimpleConvHead(nn.Module):
    """Simple convolution head for auxiliary branch."""

    def __init__(
        self,
        num_classes,
        input_channel=AUX_FEAT_CHANNELS,
        feat_channels=AUX_FEAT_CHANNELS,
        stacked_convs=AUX_STACKED_CONVS,
        strides=STRIDES,
        reg_max=REG_MAX,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max

        act = nn.LeakyReLU(0.1, inplace=True)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(stacked_convs):
            in_ch = input_channel if i == 0 else feat_channels
            cls_conv = nn.Conv2d(in_ch, feat_channels, 3, 1, 1, bias=False)
            cls_gn = nn.GroupNorm(32, feat_channels)
            self.cls_convs.append(nn.Sequential(cls_conv, cls_gn, act))

            reg_conv = nn.Conv2d(in_ch, feat_channels, 3, 1, 1, bias=False)
            reg_gn = nn.GroupNorm(32, feat_channels)
            self.reg_convs.append(nn.Sequential(reg_conv, reg_gn, act))

        self.gfl_cls = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.gfl_reg = nn.Conv2d(feat_channels, 4 * (reg_max + 1), 3, padding=1)

        self.scales = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(len(strides))]
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.gfl_cls.weight, std=0.01)
        if self.gfl_cls.bias is not None:
            nn.init.constant_(self.gfl_cls.bias, -4.595)
        nn.init.normal_(self.gfl_reg.weight, std=0.01)
        if self.gfl_reg.bias is not None:
            nn.init.constant_(self.gfl_reg.bias, 0)

    def forward(self, feats):
        outputs = []
        for i, x in enumerate(feats):
            cls_feat, reg_feat = x, x
            for conv in self.cls_convs:
                cls_feat = conv(cls_feat)
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat)
            reg_feat = reg_feat * self.scales[i]
            outputs.append(
                torch.cat(
                    [self.gfl_cls(cls_feat), self.gfl_reg(reg_feat)], dim=1
                ).flatten(start_dim=2)
            )
        return torch.cat(outputs, dim=2).permute(0, 2, 1)


class NanoDetModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """Forward pass for FLOPs analysis (simplified, no loss computation)."""
        # Backbone
        feats = self.backbone(inputs)

        # FPN
        fpn_features = self.fpn(feats)
        aux_fpn_features = self.aux_fpn(feats)
        dual_fpn_features = tuple(
            torch.cat([f, aux_f], dim=1)
            for f, aux_f in zip(fpn_features, aux_fpn_features)
        )

        # Heads
        head_out = self.head(fpn_features)
        aux_head_out = self.aux_head(dual_fpn_features)

        return head_out, aux_head_out


# -----------------------------------------------------------------------------
# Main Model
# -----------------------------------------------------------------------------


class NanoDetPlus(BaseObjectDetection):
    """
    NanoDet-Plus: Super fast and high accuracy lightweight anchor-free object detection model
    Based on: https://github.com/RangiLyu/nanodet
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_classes = self.config.num_classes
        self.strides = STRIDES
        self.reg_max = REG_MAX
        self.detach_epoch = getattr(self.config, "detach_epoch", DETACH_EPOCH)
        self.epoch = 0
        pretrained = getattr(self.config, "pretrained", True)
        model_url = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt"

        # 1. Determine backbone initialization strategy
        # We only download ImageNet weights if pretrained=True AND no custom weights are provided
        load_backbone_pretrained = pretrained and not model_url

        # Create a proper model container with forward method (for FLOPs analysis, etc.)
        self.model = NanoDetModel()

        # Backbone
        self.model.backbone = ShuffleNetV2(
            model_size="1.0x", out_stages=(2, 3, 4), pretrain=load_backbone_pretrained
        )

        # FPN
        self.model.fpn = GhostPAN(in_channels=BACKBONE_OUT_CHANNELS)
        self.model.aux_fpn = copy.deepcopy(self.model.fpn)

        # Heads
        self.model.head = NanoDetPlusHead(num_classes=self.num_classes)
        self.model.aux_head = SimpleConvHead(num_classes=self.num_classes)

        self.model.distribution_project = Integral(self.reg_max)

        # 2. Handle full weight loading if pretrained is True
        if pretrained and model_url:
            self.load_nanodet_weights(model_url)

    def load_nanodet_weights(self, model_url):
        """
        Loads pretrained weights for NanoDet-Plus.
        """
        from torch.hub import load_state_dict_from_url

        print(f"Loading weights from {model_url}...")
        state_dict = load_state_dict_from_url(model_url, map_location=self.device)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        new_state_dict = {}
        in_channels = getattr(self.config, "in_channels", 3)
        skipped_class_mismatch = 0
        skipped_avg_model = 0
        renamed_keys = 0

        for k, v in state_dict.items():
            name = k

            # Skip avg_model keys (EMA weights, not used in this implementation)
            if name.startswith("avg_model."):
                skipped_avg_model += 1
                continue

            # Skip first conv if in_channels != 3
            if in_channels != 3 and "backbone.conv1.0.weight" in name:
                print(
                    f"Skipping {name} due to input channel mismatch ({in_channels} vs 3)"
                )
                continue

            # Skip class-specific weights if num_classes != 80 (COCO)
            if self.num_classes != 80:
                if "head.gfl_cls" in name or "aux_head.gfl_cls" in name:
                    skipped_class_mismatch += 1
                    continue

            # Rename FPN keys to match our simpler structure
            if "fpn." in name and ".blocks.0." in name:
                name = name.replace(".blocks.0.", ".")
                renamed_keys += 1

            if "aux_fpn." in name and ".blocks.0." in name:
                name = name.replace(".blocks.0.", ".")
                renamed_keys += 1

            new_state_dict[name] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)

        print(f"Loaded pretrained NanoDet weights.")
        print(
            f"  - Loaded: {len(new_state_dict) - len([k for k in new_state_dict.keys() if k in msg.missing_keys])} keys"
        )
        print(f"  - Renamed (FPN structure): {renamed_keys} keys")
        print(f"  - Skipped (class mismatch): {skipped_class_mismatch} keys")
        print(f"  - Skipped (avg_model/EMA): {skipped_avg_model} keys")

        genuinely_missing = [k for k in msg.missing_keys if k in new_state_dict]
        if genuinely_missing:
            print(f"  - Missing (unexpected): {len(genuinely_missing)} keys")
            if len(genuinely_missing) < 20:
                for k in genuinely_missing:
                    print(f"      {k}")

    def set_epoch(self, epoch):
        """Set current epoch for detach_epoch logic."""
        self.epoch = epoch

    def forward(self, inputs, targets=None):
        # Handle list of tensors (Aitlas format) by stacking into batch
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)

        # Backbone
        feats = self.model.backbone(inputs)

        # FPN
        fpn_features = self.model.fpn(feats)

        # AGM with detach logic
        if self.epoch >= self.detach_epoch:
            aux_fpn_features = self.model.aux_fpn(tuple(f.detach() for f in feats))
            dual_fpn_features = tuple(
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_features, aux_fpn_features)
            )
        else:
            aux_fpn_features = self.model.aux_fpn(feats)
            dual_fpn_features = tuple(
                torch.cat([f, aux_f], dim=1)
                for f, aux_f in zip(fpn_features, aux_fpn_features)
            )

        # Heads
        head_out = self.model.head(fpn_features)
        aux_head_out = self.model.aux_head(dual_fpn_features)

        # Decode
        batch_size = inputs.size(0)
        cls_scores, reg_preds = head_out.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )

        input_height, input_width = inputs.shape[2:]
        featmap_sizes = [
            (math.ceil(input_height / s), math.ceil(input_width / s))
            for s in self.strides
        ]
        device = inputs.device
        center_priors = torch.cat(
            [
                self.model.head.get_single_level_center_priors(
                    batch_size, featmap_sizes[i], s, dtype=inputs.dtype, device=device
                )
                for i, s in enumerate(self.strides)
            ],
            dim=1,
        )

        dis_preds = (
            self.model.distribution_project(reg_preds) * center_priors[..., 2, None]
        )
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if self.model.training and targets is not None:
            gt_meta = [
                {"boxes": t["boxes"], "labels": t["labels"], "img": inputs}
                for t in targets
            ]
            loss, loss_states = self.model.head.loss(
                head_out, gt_meta, aux_preds=aux_head_out
            )
            return {
                "classification_loss": loss_states.get("loss_qfl", loss * 0),
                "regression_loss": loss_states.get("loss_bbox", loss * 0),
                "dfl_loss": loss_states.get("loss_dfl", loss * 0),
                "aux_loss": loss_states.get("aux_loss_qfl", loss * 0)
                + loss_states.get("aux_loss_bbox", loss * 0),
                "total_loss": loss,
            }
        else:
            scores = cls_scores.sigmoid()
            final_outputs = []

            for i in range(batch_size):
                scores_per_img = scores[i]
                boxes_per_img = decoded_bboxes[i]
                mask = scores_per_img > getattr(self.config, "score_threshold", 0.05)

                if not mask.any():
                    final_outputs.append(
                        {
                            "boxes": torch.zeros((0, 4), device=inputs.device),
                            "scores": torch.zeros((0,), device=inputs.device),
                            "labels": torch.zeros((0,), device=inputs.device),
                        }
                    )
                    continue

                keep_indices = torch.where(mask)
                masked_boxes = boxes_per_img[keep_indices[0]]
                masked_scores = scores_per_img[mask]
                masked_labels = keep_indices[1]

                keep = batched_nms(
                    masked_boxes,
                    masked_scores,
                    masked_labels,
                    getattr(self.config, "nms_threshold", 0.6),
                )

                final_outputs.append(
                    {
                        "boxes": masked_boxes[keep],
                        "scores": masked_scores[keep],
                        "labels": masked_labels[keep],
                    }
                )

            return final_outputs
