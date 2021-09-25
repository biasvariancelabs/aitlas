import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .detection import BaseDetectionClassifier

class FastRCNN(BaseDetectionClassifier):
    def __init__(self, config):
        BaseDetectionClassifier.__init__(self, config)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = self.config.pretrained)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.num_classes)

        ##### TRGNI ######## VO BASE CLASSIFIER ####
        # select either GPU or CPU as the active device
        # GPU is always preferred
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # move model to the right devicee
        self.model.to(device)
        ############################################

    def forward_train(self, x, y):
        return self.model.forward(x, targets = y)

    def forward_eval(self, x):
        return self.model.forward(x)

    
