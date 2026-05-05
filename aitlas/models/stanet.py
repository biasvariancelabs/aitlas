"""STANet: A Spatio-Temporal Attention Network for Change Detection"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

# Assuming BaseChangeDetection is available in your environment
from ..base.change_detection import BaseChangeDetection


# -----------------------------------------------------------------------------
# Weights Init (from change_detection_pytorch/stanet/decoder.py)
# -----------------------------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# -----------------------------------------------------------------------------
# BAM (from change_detection_pytorch/stanet/BAM.py)
# -----------------------------------------------------------------------------
class BAM(nn.Module):
    """Basic self-attention module"""

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        print("ds: ", ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-0.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input

        return out


# -----------------------------------------------------------------------------
# PAM (from change_detection_pytorch/stanet/PAM2.py)
# -----------------------------------------------------------------------------
class _PAMBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input/Output:
        N * C  * H  * (2*W)
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to partition the input feature maps
        ds                : downsampling scale
    """

    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(_PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.key_channels),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.key_channels),
        )
        self.f_value = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.value_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)
        # input shape: b,c,h,2w
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2

        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)  # B*N*H*W*2
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)  # B*N*H*W*2
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)  # B*N*H*W*2

        local_block_cnt = 2 * self.scale * self.scale

        #  self-attention func
        def func(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
            sim_map = (self.key_channels**-0.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            context_local = context_local.view(
                batch_size_new, self.value_channels, h_local, w_local, 2
            )
            return context_local

        #  Parallel Computing to speed up
        v_list = [
            value[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            for i in range(0, local_block_cnt, 2)
        ]
        v_locals = torch.cat(v_list, dim=0)
        q_list = [
            query[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            for i in range(0, local_block_cnt, 2)
        ]
        q_locals = torch.cat(q_list, dim=0)
        k_list = [
            key[:, :, local_x[i] : local_x[i + 1], local_y[i] : local_y[i + 1]]
            for i in range(0, local_block_cnt, 2)
        ]
        k_locals = torch.cat(k_list, dim=0)

        context_locals = func(v_locals, q_locals, k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size * (j + i * self.scale)
                right = batch_size * (j + i * self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)

        if self.ds != 1:
            context = F.interpolate(context, [h * self.ds, 2 * w * self.ds])

        return context


class PAMBlock(_PAMBlock):
    def __init__(self, in_channels, key_channels=None, value_channels=None, scale=1, ds=1):
        if key_channels is None:
            key_channels = in_channels // 8
        if value_channels is None:
            value_channels = in_channels
        super(PAMBlock, self).__init__(in_channels, key_channels, value_channels, scale, ds)


class PAM(nn.Module):
    """
    PAM module
    """

    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds  # output stride
        self.value_channels = out_channels
        self.key_channels = out_channels // 8

        self.stages = nn.ModuleList(
            [
                self._make_stage(in_channels, self.key_channels, self.value_channels, size, self.ds)
                for size in sizes
            ]
        )
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                in_channels * self.group,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
        )

    def _make_stage(self, in_channels, key_channels, value_channels, size, ds):
        return PAMBlock(in_channels, key_channels, value_channels, size, ds)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(torch.cat(context, 1))
        return output


# -----------------------------------------------------------------------------
# Decoders (from change_detection_pytorch/stanet/decoder.py)
# -----------------------------------------------------------------------------
class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class BackboneDecoder(nn.Module):
    def __init__(self, fc, BatchNorm, encoder_out_channels):
        super(BackboneDecoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(encoder_out_channels[2], 96)
        self.dr3 = DR(encoder_out_channels[3], 96)
        self.dr4 = DR(encoder_out_channels[4], 96)
        self.dr5 = DR(encoder_out_channels[5], 96)
        self.last_conv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(self.fc),
            nn.ReLU(),
        )

        self._init_weight()

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):
        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)
        x = F.interpolate(x, size=x2.size()[2:], mode="bilinear", align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode="bilinear", align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CDSA(nn.Module):
    """self attention module for change detection"""

    def __init__(self, in_c, ds=1, mode="PAM"):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        self.mode = mode
        if self.mode == "BAM":
            self.Self_Att = BAM(self.in_C, ds=self.ds)
        elif self.mode == "PAM":
            self.Self_Att = PAM(
                in_channels=self.in_C,
                out_channels=self.in_C,
                sizes=[1, 2, 4, 8],
                ds=self.ds,
            )
        self.apply(weights_init)

    def forward(self, x1, x2):
        width = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:width], x[:, :, :, width:]


class STANetDecoder(nn.Module):
    def __init__(self, encoder_out_channels, f_c=64, sa_mode="PAM"):
        super(STANetDecoder, self).__init__()
        self.out_channel = f_c
        self.backbone_decoder = BackboneDecoder(f_c, nn.BatchNorm2d, encoder_out_channels)
        self.netA = CDSA(in_c=f_c, ds=1, mode=sa_mode)

    def forward(self, *features):
        # fetch feature maps
        feature_0 = features[0]
        feature_1 = features[1]

        # NOTE: In the change_detection.pytorch library it expects specific indices for ResNet features
        # [0]: input, [1]: conv1, [2]: layer1, [3]: layer2, [4]: layer3, [5]: layer4

        feature_0 = self.backbone_decoder(feature_0[5], feature_0[2], feature_0[3], feature_0[4])
        feature_1 = self.backbone_decoder(feature_1[5], feature_1[2], feature_1[3], feature_1[4])
        feature_0, feature_1 = self.netA(feature_0, feature_1)
        return feature_0, feature_1


# -----------------------------------------------------------------------------
# Segmentation Head (from change_detection_pytorch/base/heads.py)
# -----------------------------------------------------------------------------
class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=None,
        upsampling=1,
        align_corners=True,
    ):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.Upsample(scale_factor=upsampling, mode="bilinear", align_corners=align_corners)
            if upsampling > 1
            else nn.Identity()
        )
        # Simple activation check
        if activation == "sigmoid":
            act = nn.Sigmoid()
        elif activation == "softmax":
            act = nn.Softmax(dim=1)
        else:
            act = nn.Identity()

        super().__init__(conv2d, upsampling, act)


# -----------------------------------------------------------------------------
# Main STANet Class
# -----------------------------------------------------------------------------
class STANetModel(nn.Module):
    """
    Implementation of STANet: A Spatio-Temporal Attention Network for Change Detection
    Based on: https://github.com/likyoo/change_detection.pytorch
    Original paper: https://www.mdpi.com/2072-4292/12/10/1662
    DOI: 10.3390/rs12101662
    """

    def __init__(self, in_channels=3, num_classes=2, pretrained=True, return_distance_map=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.return_distance_map = return_distance_map

        # Backbone (ResNet18)
        resnet = models.resnet18(pretrained=pretrained)

        # Patch first conv layer for arbitrary input channels
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias,
            )

            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            resnet.conv1 = new_conv

        # We manually construct it to match the feature list output format of SMP
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Encoder output channels [input, conv1, layer1, layer2, layer3, layer4]
        # ResNet18: [3, 64, 64, 128, 256, 512], can be calculated with _get_encoder_channels
        self.encoder_out_channels = (3, 64, 64, 128, 256, 512)

        # Decoder
        self.decoder = STANetDecoder(
            encoder_out_channels=self.encoder_out_channels,
            sa_mode="PAM",
            f_c=64,  # Default in library
        )

        # Head
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channel * 2,
            out_channels=self.num_classes,
            activation=None,
            kernel_size=3,
        )

    def _get_encoder_channels(self):
        """Dynamically calculates channel depths"""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, 224, 224)
            features = self.forward_backbone(dummy_input)
            channels = [f.shape[1] for f in features]

        if was_training:
            self.train()
        return tuple(channels)

    def forward_backbone(self, x):
        # Emulate SMP encoder output format: list of features
        # [x, conv1_feat, layer1_feat, layer2_feat, layer3_feat, layer4_feat]
        features = [x]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # [1]

        x = self.maxpool(x)
        c2 = self.layer1(x)
        features.append(c2)  # [2]

        c3 = self.layer2(c2)
        features.append(c3)  # [3]

        c4 = self.layer3(c3)
        features.append(c4)  # [4]

        c5 = self.layer4(c4)
        features.append(c5)  # [5]

        return features

    def forward(self, x1, x2):
        # Extract features (Siamese)
        features1 = self.forward_backbone(x1)
        features2 = self.forward_backbone(x2)

        # Decode and Attention
        # Pass as two arguments, decoder handles unpacking
        feat1, feat2 = self.decoder(features1, features2)

        # Head
        # NOTE: The original STANet paper uses a distance map (L2) and batch-balanced contrastive loss for change detection.
        # We switch the default option to a logit-based head (like they do in change_detection.pytorch) to maintain
        # consistency with other models in Aitlas toolbox.
        if self.return_distance_map:
            dist = F.pairwise_distance(feat1, feat2, keepdim=True)
            dist = F.interpolate(dist, x1.shape[2:], mode="bilinear", align_corners=True)
            return dist
        else:
            decoder_output = torch.cat([feat1, feat2], dim=1)
            decoder_output = F.interpolate(
                decoder_output, size=x1.shape[2:], mode="bilinear", align_corners=True
            )
            masks = self.segmentation_head(decoder_output)

        return masks


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------


class STANet(BaseChangeDetection):
    """
    Wrapper for STANet
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = STANetModel(
            in_channels=3,  # self.config.in_channels,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            return_distance_map=False,  # self.config.return_distance_map
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)
