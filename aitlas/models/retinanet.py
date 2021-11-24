import torch
import torchvision

from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.retinanet import RetinaNetRegressionHead

from ..base.detection import BaseDetectionClassifier

class RetinaNet(BaseDetectionClassifier):
    def __init__(self, config):
        BaseDetectionClassifier.__init__(self, config)

        self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained = self.config.pretrained,
                                                                         pretrained_backbone = self.config.pretrained_backbone,
                                                                         trainable_backbone_layers = self.config.trainable_backbone_layers,
                                                                         detections_per_img = self.config.detections_per_img, 
                                                                         max_size = self.config.max_size)

        self.model.head.classification_head = RetinaNetClassificationHead(self.model.backbone.out_channels, 
                                                                          self.model.anchor_generator.num_anchors_per_location()[0], 
                                                                          config['num_classes'])
        self.model.head.regression_head = RetinaNetRegressionHead(self.model.backbone.out_channels, 
                                                                  self.model.anchor_generator.num_anchors_per_location()[0])
    
    def load_lr_scheduler(self):
        if self.config.use_dota_scheduling:
            return torch.optim.lr_scheduler.MultiStepLR(self.load_optimizer(), milestones=[0, 120000, 160000], gamma=0.1)
        else:
            return None
