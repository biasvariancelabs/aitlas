import segmentation_models_pytorch as smp

from ..base import BaseSegmentationClassifier


class Unet(BaseSegmentationClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet"
            if self.config.pretrained
            else None,  # set pretrained weights for encoder
            classes=self.config.num_classes,
        )

    def forward(self, x):
        return self.model(x)
