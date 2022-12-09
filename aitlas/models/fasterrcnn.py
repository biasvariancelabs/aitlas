from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


from ..base import BaseObjectDetection


class FasterRCNN(BaseObjectDetection):
    def __init__(self, config):
        super().__init__(config)

        # load an object detection model pre-trained on COCO
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if self.config.pretrained else None
        )

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.config.num_classes
        )

    def forward(self, inputs, targets=None):
        return self.model.forward(inputs, targets)
