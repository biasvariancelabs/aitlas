import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .detection import BaseDetectionClassifier

class FastRCNN(BaseDetectionClassifier):
    def __init__(self, config):
        BaseDetectionClassifier.__init__(self, config)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn( pretrained = True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

    def forward_train(self, x, y):
        return self.model.forward(x, targets = y)

    def forward_eval(self, x):
        return self.model.forward(x)

    
