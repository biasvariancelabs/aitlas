"""EfficientDet: Scalable and Efficient Object Detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision import models
from torchvision.ops.boxes import nms as nms_torch

from ..base import BaseObjectDetection


# -----------------------------------------------------------------------------
# Utils & Helper Functions (from src/utils.py and src/loss.py)
# -----------------------------------------------------------------------------

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


def generate_anchors(base_size=16, ratios=None, scales=None):
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.register_buffer('mean', torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)))
        else:
            self.register_buffer('mean', mean)
        if std is None:
            self.register_buffer('std', torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)))
        else:
            self.register_buffer('std', std)

    def forward(self, boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        anchors = torch.from_numpy(all_anchors.astype(np.float32)).to(image.device)
        return anchors


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0, dtype=torch.float32, device=classifications.device))
                classification_losses.append(torch.tensor(0, dtype=torch.float32, device=classifications.device))
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones(classification.shape, device=classifications.device) * -1

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape, device=classifications.device) * alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(classifications.device)
                targets = targets / norm

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0, dtype=torch.float32, device=classifications.device))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


# -----------------------------------------------------------------------------
# Layers (from src/model.py)
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5), nn.ReLU())

    def forward(self, input):
        return self.conv(input)


# -----------------------------------------------------------------------------
# EfficientNet Encoder (using torchvision)
# -----------------------------------------------------------------------------

def _map_original_to_torchvision_key(orig_key):
    """
    Map original efficientnet-pytorch backbone keys to torchvision EfficientNet keys.
    
    Original structure (efficientnet-pytorch):
        backbone_net.model._conv_stem.weight
        backbone_net.model._bn0.*
        backbone_net.model._blocks.{0-15}._{component}.*
        e.g., backbone_net.model._blocks.0._depthwise_conv.weight
    
    Torchvision structure:
        features.0.0.weight  (stem conv)
        features.0.1.*       (stem BN)
        features.{1-7}.{block_idx}.block.{sub_idx}.{param}
        e.g., features.1.0.block.0.0.weight
    
    Block mapping:
        _blocks.0  -> features.1.0.block
        _blocks.1  -> features.2.0.block
        _blocks.2  -> features.2.1.block
        _blocks.3  -> features.3.0.block
        _blocks.4  -> features.3.1.block
        _blocks.5  -> features.4.0.block
        _blocks.6  -> features.4.1.block
        _blocks.7  -> features.4.2.block
        _blocks.8  -> features.5.0.block
        _blocks.9  -> features.5.1.block
        _blocks.10 -> features.5.2.block
        _blocks.11 -> features.6.0.block
        _blocks.12 -> features.6.1.block
        _blocks.13 -> features.6.2.block
        _blocks.14 -> features.6.3.block
        _blocks.15 -> features.7.0.block
    
    Component mapping within each block:
        _depthwise_conv.weight -> block.X.0.0.weight
        _bn1.*                 -> block.X.0.1.*
        _se_reduce.*           -> block.X.1.fc1.*
        _se_expand.*           -> block.X.1.fc2.*
        _project_conv.weight   -> block.X.2.0.weight
        _bn2.*                 -> block.X.2.1.*
        _expand_conv.weight    -> block.X.3.0.weight
        _bn0.*                 -> block.X.3.1.*
    """
    # Remove 'backbone_net.' prefix if present
    if orig_key.startswith('backbone_net.'):
        orig_key = orig_key.replace('backbone_net.', '')
    
    # Stem conv: model._conv_stem.weight -> features.0.0.weight
    if 'model._conv_stem' in orig_key:
        return 'features.0.0.' + orig_key.split('model._conv_stem.')[-1]
    
    # Stem BN: model._bn0.* -> features.0.1.*
    if 'model._bn0' in orig_key:
        suffix = orig_key.split('model._bn0.')[-1]
        return f'features.0.1.{suffix}'
    
    # Block mapping: model._blocks.{idx}._{component}.{param}
    if 'model._blocks.' in orig_key:
        # Extract parts after 'model._blocks.'
        parts = orig_key.split('model._blocks.')
        if len(parts) < 2:
            return None
        
        rest = parts[1]  # e.g., "0._depthwise_conv.weight"
        rest_parts = rest.split('.', 2)  # Split into at most 3 parts
        if len(rest_parts) < 3:
            return None
        
        block_idx = int(rest_parts[0])  # e.g., 0
        component = rest_parts[1]  # e.g., "_depthwise_conv" (already has underscore)
        param = rest_parts[2]  # e.g., "weight"
        
        # Map block index to (features_idx, block_in_features_idx)
        block_mapping = {
            0: (1, 0),
            1: (2, 0),
            2: (2, 1),
            3: (3, 0),
            4: (3, 1),
            5: (4, 0),
            6: (4, 1),
            7: (4, 2),
            8: (5, 0),
            9: (5, 1),
            10: (5, 2),
            11: (6, 0),
            12: (6, 1),
            13: (6, 2),
            14: (6, 3),
            15: (7, 0),
        }
        
        if block_idx not in block_mapping:
            return None
        
        feat_idx, blk_idx = block_mapping[block_idx]
        
        # Map component to torchvision sub-block indices
        # Torchvision MBConv structure differs from efficientnet-pytorch:
        #
        # For blocks WITHOUT skip connection (block 0):
        #   Original: depthwise → BN1 → project → BN2 → SE
        #   Torchvision: depthwise → BN → SE → project → BN
        #   - block.0.* = depthwise conv + BN
        #   - block.1.* = SE (fc1, fc2)
        #   - block.2.* = project conv + BN
        #
        # For blocks WITH skip connection (blocks 1-15):
        #   Original: expand → BN0 → depthwise → BN1 → SE → project → BN2
        #   Torchvision: expand → BN → depthwise → BN → SE → project → BN
        #   - block.0.* = expand conv + BN
        #   - block.1.* = depthwise conv + BN
        #   - block.2.* = SE (fc1, fc2)
        #   - block.3.* = project conv + BN
        
        # Determine if this block has skip connection (blocks 1-15 have it)
        has_skip = block_idx > 0
        
        if has_skip:
            # Blocks with skip connection (1-15)
            component_map = {
                '_expand_conv': (0, 0, 'weight'),    # -> block.X.0.0.weight
                '_bn0': (0, 1, param),                # -> block.X.0.1.{param}
                '_depthwise_conv': (1, 0, 'weight'),  # -> block.X.1.0.weight
                '_bn1': (1, 1, param),                # -> block.X.1.1.{param}
                '_se_reduce': (2, 'fc1', param),      # -> block.X.2.fc1.{param}
                '_se_expand': (2, 'fc2', param),      # -> block.X.2.fc2.{param}
                '_project_conv': (3, 0, 'weight'),    # -> block.X.3.0.weight
                '_bn2': (3, 1, param),                # -> block.X.3.1.{param}
            }
        else:
            # Block 0 (no skip connection)
            component_map = {
                '_depthwise_conv': (0, 0, 'weight'),  # -> block.X.0.0.weight
                '_bn1': (0, 1, param),                 # -> block.X.0.1.{param}
                '_se_reduce': (1, 'fc1', param),       # -> block.X.1.fc1.{param}
                '_se_expand': (1, 'fc2', param),       # -> block.X.1.fc2.{param}
                '_project_conv': (2, 0, 'weight'),     # -> block.X.2.0.weight
                '_bn2': (2, 1, param),                 # -> block.X.2.1.{param}
            }
        
        if component not in component_map:
            return None
        
        sub_idx, sub_comp, sub_param = component_map[component]
        
        return f'features.{feat_idx}.{blk_idx}.block.{sub_idx}.{sub_comp}.{sub_param}'
    
    return None


def _load_original_weights_to_torchvision(tv_model, state_dict, verbose=True):
    """
    Load original efficientnet-pytorch weights into torchvision EfficientNet model.
    
    Args:
        tv_model: torchvision EfficientNet model
        state_dict: state dict from original efficientnet-pytorch model
        verbose: print loading progress
    
    Returns:
        dict with statistics about loaded weights
    """
    tv_state = tv_model.state_dict()
    new_state = {}
    
    matched = 0
    skipped = 0
    missing_original = []
    
    # Map original keys to torchvision keys
    for orig_key, orig_value in state_dict.items():
        if not orig_key.startswith('backbone_net.'):
            continue
        
        tv_key = _map_original_to_torchvision_key(orig_key)
        
        if tv_key is None:
            if verbose:
                print(f"  Could not map: {orig_key}")
            skipped += 1
            continue
        
        if tv_key in tv_state and tv_state[tv_key].shape == orig_value.shape:
            new_state[tv_key] = orig_value
            matched += 1
        else:
            if tv_key not in tv_state:
                missing_original.append(tv_key)
                if verbose:
                    print(f"  Key not in torchvision: {tv_key} (from {orig_key})")
            elif tv_state[tv_key].shape != orig_value.shape:
                if verbose:
                    print(f"  Shape mismatch: {tv_key} {tv_state[tv_key].shape} vs {orig_value.shape}")
                skipped += 1
    
    # Load into model
    tv_state.update(new_state)
    tv_model.load_state_dict(tv_state)
    
    return {
        'matched': matched,
        'skipped': skipped,
        'missing_in_torchvision': missing_original,
    }


class EfficientNetEncoder(nn.Module):
    def __init__(self, compound_coef=0, in_channels=3, pretrained=True):
        super(EfficientNetEncoder, self).__init__()

        # Map compound_coef to the corresponding EfficientNet variant (B0-B7)
        backbone_fns = [
            models.efficientnet_b0, models.efficientnet_b1, models.efficientnet_b2,
            models.efficientnet_b3, models.efficientnet_b4, models.efficientnet_b5,
            models.efficientnet_b6, models.efficientnet_b7
        ]

        # Clamp to available versions
        compound_coef = max(0, min(compound_coef, 7))
        
        # Load with ImageNet weights initially (we'll replace them if needed)
        model = backbone_fns[compound_coef](weights='IMAGENET1K_V1' if pretrained else None)

        # Patch first conv layer if input channels != 3
        if in_channels != 3:
            old_conv = model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )
            # Copy weights if we have them
            if pretrained and in_channels == 3:
                new_conv.weight.data = old_conv.weight.data[:, :in_channels, :, :]
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            model.features[0][0] = new_conv

        # Truncate at index 8 to include all blocks (features[1] through features[7])
        # features[8] is the final head we don't need
        self.features = model.features[:8]

    def forward(self, x):
        """
        Extracts P3, P4, P5 features.
        Matches src/model.py by extracting after the first block of each stage.
        """
        feature_maps = []
        for i, layer in enumerate(self.features):
            if i in [3, 4, 6]:
                # The first block of the Sequential stage is the stride-2 block
                # We extract the feature immediately after it to match original
                x = layer[0](x)
                feature_maps.append(x)

                # We then process the rest of the stage so 'x' is ready for the next layer
                for j in range(1, len(layer)):
                    x = layer[j](x)
            else:
                x = layer(x)

        return feature_maps # P3, P4, P5


# -----------------------------------------------------------------------------
# Decoder: BiFPN, Regressor, Classifier (from src/model.py)
# -----------------------------------------------------------------------------

class BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = ConvBlock(num_channels)
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)
        self.conv7_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)
        self.p6_downsample = nn.MaxPool2d(kernel_size=2)
        self.p7_downsample = nn.MaxPool2d(kernel_size=2)

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2))
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2))
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs
        
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in))
        
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up))
        
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up))

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))
        
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out))
        
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out))
        
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        output = inputs.permute(0, 2, 3, 1)
        return output.contiguous().view(output.shape[0], -1, 4)


class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        inputs = self.act(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        output = inputs.contiguous().view(inputs.shape[0], inputs.shape[1], inputs.shape[2], self.num_anchors,
                                          self.num_classes)
        return output.contiguous().view(output.shape[0], -1, self.num_classes)


# -----------------------------------------------------------------------------
# Main EfficientDet Model Class (from src/model.py)
# -----------------------------------------------------------------------------

class EfficientDetModel(nn.Module):
    """
    Implementation of EfficientDet: Scalable and Efficient Object Detection
    Based on: https://github.com/signatrix/efficientdet
    Original paper: https://ieeexplore.ieee.org/document/9156454
    DOI: 10.1109/CVPR42600.2020.01079
    """
    def __init__(self, in_channels=3, num_anchors=9, num_classes=20, compound_coef=0, pretrained=True):
        super(EfficientDetModel, self).__init__()
        self.compound_coef = max(0, min(compound_coef, 7))

        # Config mapping from original src/model.py:
        # D_bifpn = 2 + compound_coef
        # D_class = 3 + compound_coef // 3
        W_bifpn = [64, 88, 112, 160, 224, 288, 384, 384]
        D_bifpn = [2, 3, 4, 5, 6, 7, 8, 8] # Matches 2 + compound_coef logic
        D_class = [3, 3, 3, 4, 4, 4, 5, 5] # Matches 3 + compound_coef // 3 logic

        # Backbone output channels (P3, P4, P5) for torchvision EfficientNet B0-B7
        # Note: Verified in environment to match 'start-of-stage' extraction
        backbone_out_channels = [
            [40, 80, 192],   # B0
            [40, 80, 192],   # B1
            [48, 88, 208],   # B2
            [48, 96, 232],   # B3
            [56, 112, 272],  # B4
            [64, 128, 304],  # B5
            [72, 144, 344],  # B6
            [80, 160, 384]   # B7
        ]
        
        self.num_channels = W_bifpn[self.compound_coef]
        c3_size, c4_size, c5_size = backbone_out_channels[self.compound_coef]

        # Conv layers to project backbone features to BiFPN channels
        self.conv3 = nn.Conv2d(c3_size, self.num_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(c4_size, self.num_channels, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(c5_size, self.num_channels, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(c5_size, self.num_channels, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1))

        self.bifpn = nn.Sequential(*[BiFPN(self.num_channels) for _ in range(D_bifpn[self.compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.num_channels, num_anchors=num_anchors,
                                   num_layers=D_class[self.compound_coef])
        self.classifier = Classifier(in_channels=self.num_channels, num_anchors=num_anchors, num_classes=num_classes,
                                     num_layers=D_class[self.compound_coef])

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)

        self.backbone_net = EfficientNetEncoder(compound_coef=self.compound_coef, in_channels=in_channels, pretrained=pretrained)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs, targets=None):
        if self.training and targets is not None:
            is_training = True
            img_batch = inputs
            annotations = targets
        else:
            is_training = False
            img_batch = inputs

        # Padding to make image size divisible by 128 (max stride)
        # This prevents dimension mismatch in BiFPN upsampling
        h, w = img_batch.shape[2:]
        pad_h = (128 - h % 128) % 128
        pad_w = (128 - w % 128) % 128
        if pad_h > 0 or pad_w > 0:
            img_batch = F.pad(img_batch, (0, pad_w, 0, pad_h))

        c3, c4, c5 = self.backbone_net(img_batch)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)
        p5 = self.conv5(c5)
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)

        features = [p3, p4, p5, p6, p7]
        features = self.bifpn(features)

        regression = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        if is_training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            # Return raw outputs for get_predicted processing
            return classification, transformed_anchors


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------

class EfficientDet(BaseObjectDetection):
    """
    Wrapper for EfficientDet
    """
    def __init__(self, config):
        super().__init__(config)
        
        in_channels = getattr(self.config, "in_channels", 3)
        num_classes = self.config.num_classes
        compound_coef = getattr(self.config, "compound_coef", 0)
        pretrained = self.config.pretrained
        model_url = "https://raw.githubusercontent.com/signatrix/efficientdet/master/trained_models/signatrix_efficientdet_coco.pth"

        # 1. Determine backbone initialization strategy
        # We only download ImageNet weights if pretrained=True AND no custom weights are provided
        load_backbone_pretrained = pretrained and not model_url

        self.model = EfficientDetModel(
            in_channels=in_channels,
            num_anchors=9,
            num_classes=num_classes,
            compound_coef=compound_coef,
            pretrained=load_backbone_pretrained
        )
        self.model.to(self.device)

        # 2. Handle full weight loading if pretrained is True
        if pretrained and model_url:
            self.load_efficientdet_weights(model_url)

    def load_efficientdet_weights(self, model_url):
        """
        Loads pretrained weights for EfficientDet.
        Handles full model objects saved with DataParallel by injecting fake 'src' 
        modules into sys.modules to redirect unpickling to local classes.
        """
        import os
        import sys
        from types import ModuleType
        from torch.hub import download_url_to_file, get_dir

        # 1. Prepare download path (mimic load_state_dict_from_url)
        hub_dir = get_dir()
        checkpoints_dir = os.path.join(hub_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        filename = os.path.basename(model_url)
        cached_file = os.path.join(checkpoints_dir, filename)

        # 2. Download if not exists
        if not os.path.exists(cached_file):
            print(f"Downloading weights from {model_url} to {cached_file}...")
            download_url_to_file(model_url, cached_file)
        
        # 3. Inject fake modules into sys.modules to satisfy torch.load
        # This redirects 'src.model.EfficientDet' etc. to our local classes
        fake_modules = {}
        try:
            class MockModule(ModuleType):
                def __init__(self, name):
                    super().__init__(name)
                    self.__path__ = []

                def __getattr__(self, name):
                    # Return local classes if we have them
                    if name in globals():
                        return globals()[name]
                    # Specific mappings for known name differences
                    if name == 'EfficientDet':
                        return EfficientDetModel
                    if name == 'EfficientNet':
                        return EfficientNetEncoder
                    # Fallback to a dummy class for everything else
                    class Dummy(nn.Module):
                        def __init__(self, *args, **kwargs):
                            super().__init__()
                        def forward(self, x): return x
                    return Dummy

            # Inject all modules we might need
            for mod_name in ['src', 'src.model', 'src.utils', 'src.loss', 
                            'efficientnet_pytorch', 'efficientnet_pytorch.model', 
                            'efficientnet_pytorch.utils']:
                if mod_name not in sys.modules:
                    fake_modules[mod_name] = sys.modules.get(mod_name)
                    sys.modules[mod_name] = MockModule(mod_name)
            
            print(f"Loading weights from {cached_file}...")
            # We MUST use weights_only=False because the file contains full model objects
            data = torch.load(cached_file, map_location=self.device, weights_only=False)
            
        finally:
            # Clean up fake modules to avoid side effects
            for mod_name in fake_modules:
                if fake_modules[mod_name] is None:
                    del sys.modules[mod_name]
                else:
                    sys.modules[mod_name] = fake_modules[mod_name]
        
        # Extract state_dict from the loaded data
        if isinstance(data, dict):
            state_dict = data
        elif hasattr(data, 'module'):
            # It's a DataParallel object
            state_dict = data.module.state_dict()
        else:
            # It's a full model object
            state_dict = data.state_dict()

        in_channels = getattr(self.config, "in_channels", 3)

        print(f"Loading pretrained EfficientDet weights...")
        print(f"  Target: {self.num_classes} classes, {in_channels} input channels")
        
        # 1. Load BiFPN and head weights (non-backbone) directly
        # These have the same structure in both implementations
        new_state_dict = {}
        skipped_classifier = 0
        skipped_stem = 0
        
        for k, v in state_dict.items():
            # Skip backbone keys - we'll handle them separately with mapping
            if k.startswith('backbone_net.'):
                continue
            
            # Skip classifier header if num_classes doesn't match
            if self.num_classes != 80 and "classifier.header" in k:
                skipped_classifier += 1
                continue
            
            # Skip stem conv if in_channels doesn't match
            if in_channels != 3 and "backbone_net.features.0.0.weight" in k:
                skipped_stem += 1
                continue
            
            new_state_dict[k] = v
        
        # 2. Load backbone weights using the mapping function
        print(f"\nMapping backbone weights from efficientnet-pytorch to torchvision...")
        backbone_stats = _load_original_weights_to_torchvision(
            self.model.backbone_net, 
            state_dict, 
            verbose=False
        )
        
        # Load BiFPN and head weights
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        
        # Collect statistics
        matched_non_backbone = len(new_state_dict) - len(msg.unexpected_keys)
        
        print(f"\nPretrained weights loaded.")
        print(f"  Non-backbone matched: {matched_non_backbone}")
        print(f"  Backbone matched: {backbone_stats['matched']}")
        print(f"  Backbone skipped: {backbone_stats['skipped']}")
        if skipped_classifier > 0:
            print(f"  Skipped classifier keys: {skipped_classifier} (num_classes mismatch)")
        if skipped_stem > 0:
            print(f"  Skipped stem conv: {skipped_stem} (in_channels mismatch)")
        
        # Report any missing keys in non-backbone
        other_missing = [k for k in msg.missing_keys 
                        if 'backbone_net' not in k 
                        and not any(x in k for x in ['anchors', 'regressBoxes', 'clipBoxes', 'focalLoss'])]
        
        if other_missing:
            print(f"  Other missing keys: {other_missing}")

    def forward(self, inputs, targets=None):
        # Aitlas provides inputs as a list of images and targets as a list of dicts.
        # EfficientDetModel expects a batch tensor for inputs and a batch tensor for targets.
        
        # Convert list of images to batch tensor
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).to(self.device)
            
        if self.model.training and targets is not None:
            # Convert Aitlas targets to the format expected by FocalLoss
            # Find max number of objects in the batch to pad
            max_objs = max([t['boxes'].shape[0] for t in targets]) if len(targets) > 0 else 0
            
            if max_objs == 0:
                # Handle empty batch of targets if necessary, though unlikely in training
                target_tensor = torch.zeros((len(targets), 0, 5), device=self.device)
            else:
                target_tensor = torch.ones((len(targets), max_objs, 5), device=self.device) * -1
                for i, target in enumerate(targets):
                    num_objs = target['boxes'].shape[0]
                    if num_objs > 0:
                        target_tensor[i, :num_objs, :4] = target['boxes']
                        target_tensor[i, :num_objs, 4] = target['labels']
            
            classification_loss, regression_loss = self.model(inputs, target_tensor)
            
            # Aitlas BaseObjectDetection.train_epoch expects a dict of losses
            return {
                "classification_loss": classification_loss,
                "regression_loss": regression_loss
            }
        else:
            classification, transformed_anchors = self.model(inputs)
            
            # Process output to match Aitlas expectations (list of dicts)
            # classification shape: [B, N, num_classes]
            # transformed_anchors shape: [B, N, 4]
            
            final_outputs = []
            for i in range(classification.shape[0]):
                # Get max score and label for each anchor
                scores, labels = torch.max(classification[i], dim=1)
                
                # Filter by threshold
                mask = scores > 0.01
                if not mask.any():
                    final_outputs.append({
                        "boxes": torch.zeros((0, 4), device=self.device),
                        "scores": torch.zeros((0,), device=self.device),
                        "labels": torch.zeros((0,), device=self.device)
                    })
                    continue

                masked_boxes = transformed_anchors[i][mask]
                masked_scores = scores[mask]
                masked_labels = labels[mask]

                # Perform NMS
                keep = nms(torch.cat([masked_boxes, masked_scores.unsqueeze(1)], dim=1), 0.5)
                
                final_outputs.append({
                    "boxes": masked_boxes[keep],
                    "scores": masked_scores[keep],
                    "labels": masked_labels[keep]
                })
                
            return final_outputs