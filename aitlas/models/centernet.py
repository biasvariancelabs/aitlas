"""CenterNet: Objects as Points"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..base import BaseObjectDetection


# -----------------------------------------------------------------------------
# Utils & Helper Functions (from src/lib/utils/image.py and src/lib/models/utils.py)
# -----------------------------------------------------------------------------

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1, device='cpu'):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = torch.meshgrid(torch.arange(-m, m + 1, device=device), 
                          torch.arange(-n, n + 1, device=device), indexing='ij')
    h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6, device=heatmap.device)
    
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
      
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _sigmoid(x):
    y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1-1e-4)
    return y


# -----------------------------------------------------------------------------
# Decoding (from src/lib/models/decode.py)
# -----------------------------------------------------------------------------

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(heat, wh, reg, down_ratio, K=100):
    batch, cat, height, width = heat.size()

    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
        
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).long()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    
    # Scale boxes back to original input image size
    bboxes *= down_ratio

    # Pack in AiTLAS expectations (List of Dicts)
    results = []
    for b in range(batch):
        # We optionally filter boxes by a minimum confidence score
        mask = scores[b, :, 0] > 0.05
        if not mask.any():
            results.append({
                "boxes": torch.zeros((0, 4), device=heat.device),
                "scores": torch.zeros((0,), device=heat.device),
                "labels": torch.zeros((0,), dtype=torch.long, device=heat.device)
            })
        else:
            results.append({
                "boxes": bboxes[b][mask],
                "scores": scores[b, :, 0][mask],
                "labels": clses[b, :, 0][mask]
            })
    
    return results


# -----------------------------------------------------------------------------
# Loss Functions (from src/lib/models/losses.py)
# -----------------------------------------------------------------------------

def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss(nn.Module):
    """nn.Module wrapper for focal loss"""
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, out, target):
        return _neg_loss(out, target)

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


# -----------------------------------------------------------------------------
# Model (from src/lib/models/networks/msra_resnet.py)
# -----------------------------------------------------------------------------

class PoseResNet(nn.Module):
    def __init__(self, backbone_name, heads, head_conv, pretrained=True, in_channels=3):
        super(PoseResNet, self).__init__()
        self.deconv_with_bias = False
        self.heads = heads

        # Load predefined ResNet architecture from torchvision
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        
        self.conv1 = backbone.conv1
        
        # Modify the first layer if in_channels is not 3
        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels,
                self.conv1.out_channels,
                kernel_size=self.conv1.kernel_size,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
                bias=self.conv1.bias is not None
            )

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if backbone_name in ['resnet18', 'resnet34']:
            self.inplanes = 512
        else:
            self.inplanes = 2048

        # Add 3 Deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0)
                )
            else:
                fc = nn.Conv2d(256, num_output, kernel_size=1, stride=1, padding=0)
            self.__setattr__(head, fc)
        
        self._init_weights()

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for head in self.heads:
            final_layer = self.__getattr__(head)
            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    if m.weight.shape[0] == self.heads[head]:
                        if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]


# -----------------------------------------------------------------------------
# AiTLAS Wrapper
# -----------------------------------------------------------------------------

class CenterNet(BaseObjectDetection):
    """
    Implementation of CenterNet: Objects as Points
    Based on: https://github.com/xingyizhou/CenterNet
    Original paper: https://arxiv.org/abs/1904.07850
    DOI: 10.48550/arXiv.1904.07850
    """
    def __init__(self, config):
        super().__init__(config)

        in_channels=3 #self.config.in_channels
        pretrained=self.config.pretrained
        self.num_classes = self.config.num_classes
        self.max_objs = 100
        self.down_ratio = 4
        self.hm_weight = 1.0
        self.wh_weight = 0.1
        self.off_weight = 1.0
        
        heads = {'hm': self.num_classes, 'wh': 2, 'reg': 2}
        backbone_name = 'resnet50'
        head_conv = 64
        
        self.model = PoseResNet(
            backbone_name, 
            heads, 
            head_conv, 
            pretrained=pretrained, 
            in_channels=in_channels
        )
        self.model.to(self.device)

        # Loss Functions
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()

    def generate_targets(self, targets, output_h, output_w, device):
        batch_size = len(targets)
        hm = torch.zeros((batch_size, self.num_classes, output_h, output_w), device=device)
        wh = torch.zeros((batch_size, self.max_objs, 2), device=device)
        reg = torch.zeros((batch_size, self.max_objs, 2), device=device)
        ind = torch.zeros((batch_size, self.max_objs), dtype=torch.int64, device=device)
        reg_mask = torch.zeros((batch_size, self.max_objs), dtype=torch.float32, device=device)

        for b in range(batch_size):
            boxes = targets[b]['boxes']
            labels = targets[b]['labels']
            num_objs = min(boxes.shape[0], self.max_objs)
            
            for k in range(num_objs):
                bbox = boxes[k] / self.down_ratio
                cls_id = labels[k].long()
                
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    
                    ct = torch.tensor([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], device=device)
                    ct_int = ct.int()
                    
                    draw_umich_gaussian(hm[b, cls_id], ct_int, radius)
                    
                    wh[b, k] = torch.tensor([w, h], device=device)
                    ind[b, k] = ct_int[1] * output_w + ct_int[0]
                    reg[b, k] = ct - ct_int
                    reg_mask[b, k] = 1

        return hm, wh, reg, ind, reg_mask

    def forward(self, inputs, targets=None):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs).to(self.device)
            
        if self.model.training and targets is not None:
            # Gather output dimensions
            output_h = inputs.shape[2] // self.down_ratio
            output_w = inputs.shape[3] // self.down_ratio
            
            # Ground-Truth Creation (From CenterNet ctdet dataset logic)
            hm, wh, reg, ind, reg_mask = self.generate_targets(targets, output_h, output_w, self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            output = outputs[0]
            output['hm'] = _sigmoid(output['hm'])
            
            # Compute Losses
            hm_loss = self.crit(output['hm'], hm)
            wh_loss = self.crit_wh(output['wh'], reg_mask, ind, wh)
            off_loss = self.crit_reg(output['reg'], reg_mask, ind, reg)
            
            classification_loss = self.hm_weight * hm_loss
            regression_loss = self.wh_weight * wh_loss + self.off_weight * off_loss
            
            return {
                "classification_loss": classification_loss,
                "regression_loss": regression_loss
            }
        else:
            # Inference pass
            outputs = self.model(inputs)
            output = outputs[0]
            
            # Decode the CenterNet heatmap back to AiTLAS-expected standard predictions
            results = ctdet_decode(
                output['hm'], 
                output['wh'], 
                output['reg'], 
                down_ratio=self.down_ratio, 
                K=self.max_objs
            )
            return results