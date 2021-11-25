import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..base import BaseSegmentationClassifier


class FastRCNN(BaseSegmentationClassifier):
    def __init__(self, config):
        super().__init__(config)

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
