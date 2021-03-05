"""
Notes
-----
    Based on the implementation at:
        https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/blob/master/1-zbigniewwojna/main.py#L178
"""
import torch
import torch.nn as nn
from geffnet.conv2d_layers import select_conv2d
from geffnet.efficientnet_builder import round_channels, initialize_weight_goog, initialize_weight_default, \
    EfficientNetBuilder, decode_arch_def, BN_EPS_TF_DEFAULT, resolve_bn_args
from torch.hub import load_state_dict_from_url

from aitlas.base import BaseSegmentationClassifier
from aitlas.models.schemas import UNetEfficientNetModelSchema


class GenEfficientNet(nn.Module):
    def __init__(self, block_args, num_classes=1000, in_channels=3, num_features=1280, stem_size=32, fix_stem=False,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 pad_type='', act_layer=nn.ReLU, drop_connect_rate=0., se_kwargs=None, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, weight_init='goog'):
        super(GenEfficientNet, self).__init__()
        stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_channels, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        builder = EfficientNetBuilder(channel_multiplier, channel_divisor, channel_min, pad_type, act_layer, se_kwargs,
                                      norm_layer, norm_kwargs, drop_connect_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.conv_head = select_conv2d(builder.in_chs, num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        for module in self.modules():
            if weight_init == 'goog':
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
        BaseSegmentationClassifier.__init__(self, config)
        dec_ch = [32, 64, 128, 256, 1024]
        # Placeholder variables to avoid "might be referenced before assignment" warnings
        enc_ch = list()
        depth_multiplier, channel_multiplier = 1, 1
        url = ""
        if self.config.net == 'b4':
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ns-d6313a46.pth'
            enc_ch = [24, 32, 56, 160, 1792]
        if self.config.net == 'b5':
            channel_multiplier = 1.6
            depth_multiplier = 2.2
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ns-6f26d0cf.pth'
            enc_ch = [24, 40, 64, 176, 2048]
        if self.config.net == 'b6':
            channel_multiplier = 1.8
            depth_multiplier = 2.6
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ns-51548356.pth'
            enc_ch = [32, 40, 72, 200, 2304]
        if self.config.net == 'b7':
            channel_multiplier = 2.0
            depth_multiplier = 3.1
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth'
            enc_ch = [32, 48, 80, 224, 2560]
        if self.config.net == 'l2':
            channel_multiplier = 4.3
            depth_multiplier = 5.3
            url = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_l2_ns-df73bb44.pth'
            enc_ch = [72, 104, 176, 480, 5504]
        if self.config.stride == 16:
            dec_ch[4] = enc_ch[4]
        elif self.config.stride == 8:
            dec_ch[3] = enc_ch[4]

        def mod(cin, cout, k=3):
            return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2), nn.ReLU(inplace=True))

        self.bot0extra = mod(206, enc_ch[4])
        self.bot1extra = mod(206, dec_ch[4])
        self.bot2extra = mod(206, dec_ch[3])
        self.bot3extra = mod(206, dec_ch[2])
        self.bot4extra = mod(206, dec_ch[1])
        self.bot5extra = mod(206, 6)

        self.dec0 = mod(enc_ch[4], dec_ch[4])
        self.dec1 = mod(dec_ch[4], dec_ch[3])
        self.dec2 = mod(dec_ch[3], dec_ch[2])
        self.dec3 = mod(dec_ch[2], dec_ch[1])
        self.dec4 = mod(dec_ch[1], dec_ch[0])

        self.bot0 = mod(enc_ch[3] + dec_ch[4], dec_ch[4])
        self.bot1 = mod(enc_ch[2] + dec_ch[3], dec_ch[3])
        self.bot2 = mod(enc_ch[1] + dec_ch[2], dec_ch[2])
        self.bot3 = mod(enc_ch[0] + dec_ch[1], dec_ch[1])

        self.up = nn.Upsample(scale_factor=2)
        self.upps = nn.PixelShuffle(upscale_factor=2)
        self.final = nn.Conv2d(dec_ch[0], 6, 1)
        self._initialize_weights()
        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25']
        ]
        enc = GenEfficientNet(in_channels=3, block_args=decode_arch_def(arch_def, depth_multiplier),
                              num_features=round_channels(1280, channel_multiplier, 8, None), stem_size=32,
                              channel_multiplier=channel_multiplier,
                              norm_kwargs=resolve_bn_args({'bn_eps': BN_EPS_TF_DEFAULT}), pad_type='same')
        state_dict = load_state_dict_from_url(url)
        enc.load_state_dict(state_dict, strict=True)
        stem_size = round_channels(32, channel_multiplier, 8, None)
        conv_stem = select_conv2d(4, stem_size, 3, stride=2, padding='same')
        _w = enc.conv_stem.state_dict()
        _w['weight'] = torch.cat([_w['weight'], _w['weight'][:, 1:2]], 1)
        conv_stem.load_state_dict(_w)
        self.enc0 = nn.Sequential(conv_stem, enc.bn1, enc.act1, enc.blocks[0])
        self.enc1 = nn.Sequential(enc.blocks[1])
        self.enc2 = nn.Sequential(enc.blocks[2])
        self.enc3 = nn.Sequential(enc.blocks[3], enc.blocks[4])
        self.enc4 = nn.Sequential(enc.blocks[5], enc.blocks[6], enc.conv_head, enc.bn2, enc.act2)

    def forward(self, x, strip, direction, coord):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        ex = torch.cat([strip, direction, coord], 1)
        x = enc4
        if self.config.stride == 32:
            x = self.dec0(self.up(x + self.bot0extra(ex)))
            x = torch.cat([x, enc3], dim=1)
            x = self.bot0(x)
        if self.config.stride == 32 or self.config.stride == 16:
            x = self.dec1(self.up(x + self.bot1extra(ex)))
            x = torch.cat([x, enc2], dim=1)
            x = self.bot1(x)
        x = self.dec2(self.up(x))
        x = torch.cat([x, enc1], dim=1)
        x = self.bot2(x)
        x = self.dec3(self.up(x))
        x = torch.cat([x, enc0], dim=1)
        x = self.bot3(x)
        x = self.dec4(self.up(x))
        x = self.final(x)
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
