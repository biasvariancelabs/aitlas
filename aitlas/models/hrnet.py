import timm
import torch

from ..base import BaseSegmentationClassifier


class BaseHRNet(BaseSegmentationClassifier):
    model_name = ""

    def __init__(self, config):
        super().__init__(config)

        if self.config.pretrained:
            self.model = timm.create_model(
                self.model_name, pretrained=self.config.pretrained
            )
        else:
            self.model = timm.create_model(self.model_name)

    def forward(self, x):
        return self.model.forward(x)


class HRNetW18(BaseHRNet):
    model_name = "hrnet_w18"


class HRNetW30(BaseHRNet):
    model_name = "hrnet_w30"


class HRNetW32(BaseHRNet):
    model_name = "hrnet_w32"


class HRNetW40(BaseHRNet):
    model_name = "hrnet_w40"


class HRNetW44(BaseHRNet):
    model_name = "hrnet_w44"


class HRNetW48(BaseHRNet):
    model_name = "hrnet_w48"


class HRNetW64(BaseHRNet):
    model_name = "hrnet_w64"
