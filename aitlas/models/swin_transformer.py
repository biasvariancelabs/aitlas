import torch.nn as nn
import torchvision.models as models

from ..base import BaseMulticlassClassifier, BaseMultilabelClassifier


class SwinTransformer(BaseMulticlassClassifier):
    """
    A Swin Transformer V2 implementation for multi-class classification tasks. Based on <https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_v2_s.html#torchvision.models.swin_v2_s>
    """

    name = "SwinTransformerV2"

    def __init__(self, config):

        """
        Initialize a SwinTransformer object with the given configuration.

        Args:
            config (Config): A configuration object containing model-related settings.
        """
        super().__init__(config)

        self.model = models.swin_v2_s(
            weights=models.Swin_V2_S_Weights.IMAGENET1K_V1
            if self.config.pretrained
            else None,
            progress=False,
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

        if self.config.freeze:
            self.freeze()

    def freeze(self):
        """
        Freeze all the layers in the model except for the head.
        This prevents the gradient computation for the frozen layers during backpropagation.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes).
        """
        return self.model(x)


class SwinTransformerMultilabel(BaseMultilabelClassifier):
    """
    A Swin Transformer V2 implementation for multi-label classification tasks. Based on <https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_v2_s.html#torchvision.models.swin_v2_s>
    """

    name = "SwinTransformerV2"

    def __init__(self, config):
        """
        Initialize a SwinTransformerMultilabel object with the given configuration.

        Args:
            config (Config): A configuration object containing model-related settings.
        """
        super().__init__(config)

        self.model = models.swin_v2_s(
            weights=models.Swin_V2_S_Weights.IMAGENET1K_V1
            if self.config.pretrained
            else None,
            progress=False,
        )
        self.model.head = nn.Linear(
            in_features=768, out_features=self.config.num_classes, bias=True
        )

        if self.config.freeze:
            self.freeze()

    def freeze(self):
        """
        Freeze all the layers in the model except for the head.
        This prevents the gradient computation for the frozen layers during backpropagation.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes).
        """
        return self.model(x)
