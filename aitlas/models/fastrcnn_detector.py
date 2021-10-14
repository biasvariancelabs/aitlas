import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..base.detection import BaseDetectionClassifier

class FastRCNN(BaseDetectionClassifier):
    def __init__(self, config):
        BaseDetectionClassifier.__init__(self, config)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = self.config.pretrained)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)