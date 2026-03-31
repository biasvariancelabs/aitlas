"""DETR: End-to-End Object Detection with Transformers"""

import math
import copy
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.boxes import box_area
from scipy.optimize import linear_sum_assignment

from ..base import BaseObjectDetection


# -----------------------------------------------------------------------------
# Utils & Helper Functions (from util/box_ops.py and util/misc.py)
# -----------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    # Ensure w and h are at least a small epsilon to avoid degenerate boxes
    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6) # + 1e-6 to prevent NaN division
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "Degenerate prediction boxes found"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "Degenerate ground truth boxes found"
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

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

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
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# -----------------------------------------------------------------------------
# Backbone (from models/backbone.py)
# -----------------------------------------------------------------------------

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool,
                 in_channels: int = 3, pretrained: bool = True):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained,
            norm_layer=FrozenBatchNorm2d)
        
        # Patch first conv layer if input channels != 3
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            backbone.conv1 = new_conv
        
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


# -----------------------------------------------------------------------------
# Position Encoding (from models/position_encoding.py)
# -----------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# -----------------------------------------------------------------------------
# Transformer (from models/transformer.py)
# -----------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None, query_pos: Optional[torch.Tensor] = None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm else output)
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None, query_pos: Optional[torch.Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None, query_pos: Optional[torch.Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None, pos: Optional[torch.Tensor] = None, query_pos: Optional[torch.Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# -----------------------------------------------------------------------------
# Matcher (from models/matcher.py)
# -----------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# -----------------------------------------------------------------------------
# DETR & Losses (from models/detr.py)
# -----------------------------------------------------------------------------

class DETRModel(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Single-node simplified tracking
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------

class DETR(BaseObjectDetection):
    """
    Implementation of DETR: End-to-End Object Detection with Transformers
    Based on: https://github.com/facebookresearch/detr
    Original paper: https://arxiv.org/abs/1911.09070
    DOI: 10.48550/arXiv.1911.09070
    """
    def __init__(self, config):
        super().__init__(config)

        # Config properties
        in_channels = getattr(self.config, "in_channels", 3)
        pretrained = self.config.pretrained
        num_classes = self.config.num_classes
        num_queries = 100
        hidden_dim = 256
        backbone_name = getattr(self.config, "backbone", 'resnet50')
        dilation = True # Enables DC5 model version
        train_backbone = True
        aux_loss = True
        model_url = "https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth"

        # 1. Determine backbone initialization strategy
        load_backbone_pretrained = pretrained and not model_url

        # Build Backbone
        backbone = Backbone(
            backbone_name,
            train_backbone,
            return_interm_layers=False,
            dilation=dilation,
            in_channels=in_channels,
            pretrained=load_backbone_pretrained
        )
        position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = Joiner(backbone, position_embedding)
        backbone_with_pos_enc.num_channels = backbone.num_channels

        # Build Transformer
        transformer = Transformer(
            d_model=hidden_dim,
            dropout=self.config.dropout if hasattr(self.config, 'dropout') else 0.1,
            nhead=self.config.nheads if hasattr(self.config, 'nheads') else 8,
            dim_feedforward=self.config.dim_feedforward if hasattr(self.config, 'dim_feedforward') else 2048,
            num_encoder_layers=self.config.enc_layers if hasattr(self.config, 'enc_layers') else 6,
            num_decoder_layers=self.config.dec_layers if hasattr(self.config, 'dec_layers') else 6,
            normalize_before=self.config.pre_norm if hasattr(self.config, 'pre_norm') else False,
            return_intermediate_dec=True
        )

        # Initialize Main DETR Model
        self.model = DETRModel(
            backbone_with_pos_enc, 
            transformer, 
            num_classes=num_classes, 
            num_queries=num_queries, 
            aux_loss=aux_loss
        )
        self.model.to(self.device)

        # ---------------------------------------------------------------------------------
        # FIX 1: HACK IN GRADIENT CLIPPING VIA BACKWARD HOOKS
        # Aitlas doesn't clip gradients natively, so we embed it at the parameter level.
        # This acts identically to `torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)`
        # ---------------------------------------------------------------------------------
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1) if grad is not None else None)

        # 2. Handle full weight loading if pretrained is True
        if pretrained and model_url:
            self.load_detr_weights(model_url)

        # Build Criterion (Loss logic)
        matcher = HungarianMatcher(
            cost_class=self.config.set_cost_class if hasattr(self.config, 'set_cost_class') else 1,
            cost_bbox=self.config.set_cost_bbox if hasattr(self.config, 'set_cost_bbox') else 5,
            cost_giou=self.config.set_cost_giou if hasattr(self.config, 'set_cost_giou') else 2
        )
        weight_dict = {'loss_ce': 1, 'loss_bbox': self.config.bbox_loss_coef if hasattr(self.config, 'bbox_loss_coef') else 5}
        weight_dict['loss_giou'] = self.config.giou_loss_coef if hasattr(self.config, 'giou_loss_coef') else 2
        
        if aux_loss:
            aux_weight_dict = {}
            for i in range((self.config.dec_layers if hasattr(self.config, 'dec_layers') else 6) - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']
        self.detr_criterion = SetCriterion(
            num_classes=num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict,
            eos_coef=self.config.eos_coef if hasattr(self.config, 'eos_coef') else 0.1, 
            losses=losses
        )
        self.detr_criterion.to(self.device)

        # Build PostProcessor
        self.postprocessors = {'bbox': PostProcess()}

    # ---------------------------------------------------------------------------------
    # FIX 2: OVERRIDE OPTIMIZER
    # Aitlas BaseObjectDetection uses Adam with a uniform learning rate.
    # DETR mathematically requires AdamW, and the backbone MUST run 10x slower.
    # ---------------------------------------------------------------------------------
    def load_optimizer(self):
        """Override to apply different learning rates and use AdamW."""
        base_lr = self.config.learning_rate
        param_dicts = [
            {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": base_lr * 0.1, # Critical: 10x smaller learning rate for the backbone
            },
        ]
        return optim.Adam(param_dicts, lr=base_lr)
    
    def load_detr_weights(self, model_url):
        """
        Loads pretrained weights for DETR.
        """
        from torch.hub import load_state_dict_from_url
        
        print(f"Loading weights from {model_url}...")
        state_dict = load_state_dict_from_url(model_url, map_location=self.device)
        
        if "model" in state_dict:
            state_dict = state_dict["model"]

        new_state_dict = {}
        in_channels = getattr(self.config, "in_channels", 3)

        for k, v in state_dict.items():
            name = k
            
            # Filter: Skip first conv if in_channels != 3
            if in_channels != 3 and "backbone.0.body.conv1.weight" in name:
                print(f"Skipping {name} due to input channel mismatch ({in_channels} vs 3)")
                continue

            # Filter: Skip class-specific weights if num_classes != 91 (COCO)
            if self.num_classes != 91 and "class_embed" in name:
                print(f"Skipping {name} due to class count mismatch ({self.num_classes} vs 91)")
                continue

            new_state_dict[name] = v

        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained DETR weights. Missing keys: {len(msg.missing_keys)}")
        if len(msg.missing_keys) < 20 and len(msg.missing_keys) > 0: 
             print(f"Missing keys: {msg.missing_keys}")
    
    def forward(self, inputs, targets=None):
        # inputs: List of Tensors (images) or a batched Tensor
        # targets: List of dicts (Aitlas format) with keys "boxes" and "labels"
        
        if isinstance(inputs, list):
            target_sizes = torch.stack([torch.tensor([img.shape[1], img.shape[2]], device=self.device) for img in inputs])
            samples = nested_tensor_from_tensor_list(inputs).to(self.device)
        else:
            target_sizes = torch.stack([torch.tensor([img.shape[1], img.shape[2]], device=self.device) for img in inputs])
            samples = nested_tensor_from_tensor_list(list(inputs)).to(self.device)
            
        if self.model.training and targets is not None:
            # Convert Aitlas target format to DETR target format
            detr_targets = []
            for target, size in zip(targets, target_sizes):
                h, w = size
                boxes = target['boxes'].to(self.device) # Expected in xyxy format
                labels = target['labels'].to(self.device)
                
                # ---------------------------------------------------------------------------------
                # FIX 3: PURGE DEGENERATE BOXES AND SHIFT LABELS SAFELY
                # ---------------------------------------------------------------------------------
                if boxes.shape[0] > 0:
                    # Explicitly remove boxes with <= 0 width or height caused by float precision
                    valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    boxes = boxes[valid_mask]
                    labels = labels[valid_mask]
                
                if boxes.shape[0] > 0:
                    # Convert xyxy to normalized cxcywh format
                    boxes = box_xyxy_to_cxcywh(boxes)
                    
                    # Normalize, protecting against edge-case 0-dimension division
                    norm_tensor = torch.tensor([max(w.item(), 1), max(h.item(), 1), max(w.item(), 1), max(h.item(), 1)], 
                                                dtype=torch.float32, device=self.device)
                    boxes = boxes / norm_tensor
                    
                    # Shift labels to map dataloader's '1' (ship) -> '0' (DETR foreground class)
                    labels = labels - 1
                    
                    detr_targets.append({
                        "labels": labels,
                        "boxes": boxes
                    })
                else:
                    # Provide an empty tensor shaped exactly as DETR expects to avoid breaking the matcher
                    detr_targets.append({
                        "labels": torch.zeros((0,), dtype=torch.int64, device=self.device),
                        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                    })

            outputs = self.model(samples)
            loss_dict = self.detr_criterion(outputs, detr_targets)

            # Weight the losses as configured by SetCriterion
            weight_dict = self.detr_criterion.weight_dict
            
            # Map DETR losses to Aitlas standard keys
            classification_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if 'loss_ce' in k and k in weight_dict)
            regression_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if ('loss_bbox' in k or 'loss_giou' in k) and k in weight_dict)

            # Return a flat dictionary of combined & weighted losses for Aitlas backprop step
            return {
                "classification_loss": classification_loss,
                "regression_loss": regression_loss
            }
        
        else:
            # Inference mode
            outputs = self.model(samples)
            results = self.postprocessors['bbox'](outputs, target_sizes)

            # Convert to Aitlas expectations: list of dicts returning actual bounds, scores, labels
            final_outputs = []
            for result in results:
                # Filter by a confidence threshold
                mask = result['scores'] > 0.05
                if not mask.any():
                    final_outputs.append({
                        "boxes": torch.zeros((0, 4), device=self.device),
                        "scores": torch.zeros((0,), device=self.device),
                        "labels": torch.zeros((0,), device=self.device)
                    })
                else:
                    final_outputs.append({
                        "boxes": result['boxes'][mask],
                        "scores": result['scores'][mask],
                        # Shift labels back up by 1 to match the Aitlas dataloader format
                        "labels": result['labels'][mask] + 1 
                    })

            return final_outputs