"""
Notes
-----
    Based on the implementation at:
        https://github.com/CosmiQ/cresi/tree/master/cresi/net
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torch import cat

from aitlas.base import BaseSegmentationClassifier

filters = [64, 64, 128, 256, 512]


class ResnetConfig:
    """
    Helper class for the configuration of the Resnet34 encoder.
    """

    def __init__(self, num_classes, pretrained):
        """
        Parameters
        ---------
            num_classes : int
                specifies the number of classes, i.e. the channel dimension in the output mask
            pretrained : bool
                specifies whether the resnet should be pretrained or not
        """
        self.num_classes = num_classes
        self.pretrained = pretrained


class Resnet34:
    """
    Wrapper class around the Resnet34 model implementation in Pytorch extending its functionality with skip connections.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
            config : ResnetConfig
                the configuration for this model
        """
        if config.pretrained:
            self.model = models.resnet34(pretrained=config.pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, config.num_classes)
        else:
            self.model = models.resnet34(pretrained=config.pretrained,
                                         num_classes=config.num_classes)

    def forward(self, x):
        """The forward pass through the model."""
        self.model.forward(x)

    def skip_connection(self, layer):
        """
        Returns the output at the appropriate layer to be used as a skip-connection
        at the decoder part of the UNet.

        Parameters
        ----------
            layer : int, one of (0, 1, 2, 3, 4)
                specifies which layer output needs to be returned

        Returns
        -------
            result : nn.module
                the output from the model at the corresponding layer
        """
        if layer == 0:
            return nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu
            )
        elif layer == 1:
            return nn.Sequential(
                self.model.maxpool,
                self.model.layer1
            )
        elif layer == 2:
            return self.model.layer2
        elif layer == 3:
            return self.model.layer3
        elif layer == 4:
            return self.model.layer4


class ConvolutionBottleneckBlock(nn.Module):
    """
    A single block of modules to be used at the bottleneck of the `U` architecture,
    that is, between the encoder output and decoder input.
    """

    def __init__(self, in_channels, out_channels):
        """
        Parameters
        ----------
            in_channels : int
            out_channels : int
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, decoder, encoder):
        """The forward pass through the block."""
        x = cat([decoder, encoder], dim=1)
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """
    A single block of modules to be used for the decoder.
    """

    def __init__(self, in_channels, out_channels):
        """
        Parameters
        ----------
            in_channels : int
            out_channels : int
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """The forward pass through the model."""
        return self.block(x)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.contiguous()
        targets = targets.contiguous()

        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]

        outputs = torch.clamp(outputs, self.eps, 1. - self.eps)
        targets = torch.clamp(targets, self.eps, 1. - self.eps)

        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, eps=1e-6):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
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


class UNetResnet34(BaseSegmentationClassifier):
    """
    Implements a UNet-like neural network that uses a Resnet34 as an
    encoder which is the baseline solution for the SpaceNet 5 challenge.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
            config : Config
                The configuration for this model.
        """
        BaseSegmentationClassifier.__init__(self, config)
        # Number of channels of input image
        self.NUM_CHANNELS = 3
        # Specify bottleneck layers
        self.bottlenecks = nn.ModuleList([
            ConvolutionBottleneckBlock(in_channels=f * 2,
                                       out_channels=f) for f in reversed(filters[:-1])
        ]).to(self.device)
        # Specify decoder layers
        self.decoder = nn.ModuleList([
            UNetDecoderBlock(in_channels=filters[layer],
                             out_channels=filters[max(layer - 1, 0)]) for layer in range(1, len(filters))
        ]).to(self.device)
        # Specify final layers
        self.last_upsample_layer = UNetDecoderBlock(in_channels=filters[0], out_channels=filters[0] // 2).to(
            self.device)
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=filters[0] // 2,
                      out_channels=config.num_classes,
                      kernel_size=3,
                      padding=1)
        ).to(self.device)
        # Initialize bottleneck and decoder weights, encoder comes pretrained
        self.initialize_weights()
        # Set up the encoder and the skip connections
        self.encoder = Resnet34(config=ResnetConfig(num_classes=self.NUM_CHANNELS,
                                                    pretrained=config.pretrained))
        self.encoder_skip_connections = nn.ModuleList([
            self.encoder.skip_connection(layer) for layer in range(len(filters))
        ]).to(self.device)
        self.focal = FocalLoss2d()
        self.dice = DiceLoss()

    def initialize_weights(self):
        """
        Initializes the weights of the bottleneck and decoder layers.

        Weights of convolution modules are initialized with normal Kaiming He distribution.
        Weights of batch norm. modules are initialized to ones.
        All biases are set to zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(tensor=module.weight.data, nonlinearity="relu")
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        """
        The forward pass through the model.
        """
        # print(f"Input shape: {x.shape}") # (N, 3, H, W)
        encoder_results = list()
        # Pass through the encoder and save each skip connection
        for inx, encoder_stage in enumerate(self.encoder_skip_connections):
            x = encoder_stage(x)
            if inx < len(self.encoder_skip_connections) - 1:
                encoder_results.append(x.clone())
        # Pass through the bottleneck stage, using the decoder output and the skip connections
        for inx, bottleneck_stage in enumerate(self.bottlenecks):
            rev_inx = - (inx + 1)
            x = self.decoder[rev_inx](x)
            x = bottleneck_stage(x, encoder_results[rev_inx])
        # Pass through the last decoder layers
        x = self.last_upsample_layer(x)
        x = self.final_layer(x)
        # print(f"Output shape: {x.shape}") # (N, 8, H, W)
        return x

    def load_criterion(self):
        """Loads custom loss function: 25% dice coefficient + 75% focal loss"""

        def criterion(y_pred, y_true):
            return self.focal(y_pred, y_true) * 0.25 + self.dice(y_pred, y_true) * 0.75

        return criterion

    def load_optimizer(self):
        return torch.optim.Adam(self.parameters())

    def load_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(optimizer=self.load_optimizer(), milestones=[20, 25], gamma=0.2)
