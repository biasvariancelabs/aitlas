import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from ..base import BaseSegmentation
from .schemas import FastRCNNSchema


class FastRCNN(BaseSegmentation):
    schema = FastRCNNSchema

    def __init__(self, config):
        BaseSegmentation.__init__(self, config)

        # load an instance segmentation model pre-trained pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.config.num_classes
        )

    def forward(self, x):
        return self.model.forward(x)
