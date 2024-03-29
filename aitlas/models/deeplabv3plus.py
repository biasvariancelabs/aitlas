"""DeepLabV3Plus model"""
import segmentation_models_pytorch as smp

from ..base import BaseSegmentationClassifier


class DeepLabV3Plus(BaseSegmentationClassifier):
    """DeepLabV3Plus model implementation

    .. note:: Based on https://github.com/qubvel/segmentation_models.pytorch

    """

    def __init__(self, config):
        super().__init__(config)

        self.model = smp.DeepLabV3Plus(
            encoder_name="tu-xception71",
            encoder_weights="imagenet"
            if self.config.pretrained
            else None,  # set pretrained weights for encoder
            classes=self.config.num_classes,
        )

    def forward(self, x):
        return self.model(x)
