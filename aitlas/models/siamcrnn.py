"""
SiamCRNN: Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..base import BaseChangeDetection

# -----------------------------------------------------------------------------
# ConvLSTM Components (from FCN_version/deep_networks/SiamCRNN.py)
# -----------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    A multi-layer Convolutional LSTM module.
    """

    def __init__(self, input_dim, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# -----------------------------------------------------------------------------
# Main SiamCRNN Model (from FCN_version/deep_networks/SiamCRNN.py)
# -----------------------------------------------------------------------------

class SiamCRNNModel(nn.Module):
    """
    Implementation of SiamCRNN: Change Detection in Multisource VHR Images via 
    Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network (FCN Version)
    
    Based on: https://github.com/ChenHongruixuan/SiamCRNN
    Original paper: https://ieeexplore.ieee.org/document/8937755
    DOI: 10.1109/TGRS.2019.2956756
    """
    def __init__(self, in_channels=3, num_classes=2, pretrained=True):
        super(SiamCRNNModel, self).__init__()
        
        # --- Encoder: ResNet34 ---
        resnet = models.resnet34(pretrained=pretrained)
        resnet = self._convert_resnet_to_output_stride_16(resnet)
        
        # Handle input channels != 3
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias
            )
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            resnet.conv1 = new_conv

        # Decompose ResNet to access intermediate layers
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1 # 64 channels
        self.encoder_layer2 = resnet.layer2 # 128 channels
        self.encoder_layer3 = resnet.layer3 # 256 channels
        self.encoder_layer4 = resnet.layer4 # 512 channels
        
        # --- Decoder: Stacked ConvLSTMs ---
        # ConvLSTM layers for each stage (4, 3, 2, 1)
        self.convlstm_4 = ConvLSTM(input_dim=512)
        self.convlstm_3 = ConvLSTM(input_dim=256)
        self.convlstm_2 = ConvLSTM(input_dim=128)
        self.convlstm_1 = ConvLSTM(input_dim=64)
        
        # --- Smooth Layers (Fusion) ---
        self.smooth_layer_3 = self._make_smooth_layer(128)
        self.smooth_layer_2 = self._make_smooth_layer(128)
        self.smooth_layer_1 = self._make_smooth_layer(128)

        # --- Classifier ---
        self.main_clf_1 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)

    def _convert_resnet_to_output_stride_16(self, resnet):
        # Modify layer3: stride=1, dilation=2
        for n, m in resnet.layer3.named_modules():
            if 'conv1' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        # Modify layer4: stride=1, dilation=4
        for n, m in resnet.layer4.named_modules():
            if 'conv1' in n:
                m.dilation = (4, 4)
                m.padding = (4, 4)
                m.stride = (1, 1)
            elif 'conv2' in n:
                m.dilation = (4, 4)
                m.padding = (4, 4)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        return resnet

    def _make_smooth_layer(self, dim):
        return nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=dim, out_channels=dim, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

    def _forward_encoder(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)

        l1 = self.encoder_layer1(x)
        l2 = self.encoder_layer2(l1)
        l3 = self.encoder_layer3(l2)
        l4 = self.encoder_layer4(l3)
        
        return l1, l2, l3, l4

    def _upsample_add(self, x, y):
        """Upsample x to match y's size and add"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # 1. Extract Features (Siamese)
        pre_l1, pre_l2, pre_l3, pre_l4 = self._forward_encoder(pre_data)
        post_l1, post_l2, post_l3, post_l4 = self._forward_encoder(post_data)

        # 2. Stage 4 Processing
        # Concatenate along the time dimension: (B, T=2, C, H, W)
        combined_4 = torch.stack([pre_l4, post_l4], dim=1)
        _, last_state_list_4 = self.convlstm_4(combined_4)
        # Take the hidden state (h) of the first (and only) layer
        p4 = last_state_list_4[0][0] 

        # 3. Stage 3 Processing
        combined_3 = torch.stack([pre_l3, post_l3], dim=1)
        _, last_state_list_3 = self.convlstm_3(combined_3)
        p3 = last_state_list_3[0][0]
        # Fusion FPN style (Upsample previous + current)
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        # 4. Stage 2 Processing
        combined_2 = torch.stack([pre_l2, post_l2], dim=1)
        _, last_state_list_2 = self.convlstm_2(combined_2)
        p2 = last_state_list_2[0][0]
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        # 5. Stage 1 Processing
        combined_1 = torch.stack([pre_l1, post_l1], dim=1)
        _, last_state_list_1 = self.convlstm_1(combined_1)
        p1 = last_state_list_1[0][0]
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        # 6. Classification
        output = self.main_clf_1(p1)
        # Restore to original image size
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        
        return output


# -----------------------------------------------------------------------------
# Aitlas Wrapper
# -----------------------------------------------------------------------------

class SiamCRNN(BaseChangeDetection):
    """
    Wrapper for SiamCRNN
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.model = SiamCRNNModel(
            in_channels=3, #self.config.in_channels
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained
        )

    def forward(self, x1, x2):
        return self.model(x1, x2)