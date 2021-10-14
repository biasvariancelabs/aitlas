# datasets
from ..aitlas.datasets.esri_challenge import EsriChallengeDataset
from ..aitlas.datasets.dota import DotaDataset

# models
from ..aitlas.models.fastrcnn_detector import FastRCNN
from ..aitlas.models.retinanet import RetinaNet

# vizualization
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib as mpl
import numpy as np

# Define either of the two datasets:
## ESRI challenge

train_cfg = {
    "root": "E:\\AiTLAS\\aitlas\\ESRI-challenge\\data\\ESRI",
    "subset": "train",
    "subsample_percentage": 0.02,
    "batch_size": 256,
    "shuffle": True,
    "num_workers": 0,
    "transforms": ["torchvision.transforms.ToTensor"]
}
train_set = EsriChallengeDataset(train_cfg)

test_cfg = {
    "root": "E:\\AiTLAS\\aitlas\\ESRI-challenge\\data\\ESRI",
    "subset": "test",
    "subsample_percentage": 0.02,
    "batch_size": 256,
    "shuffle": True,
    "num_workers": 0,
    "transforms": ["torchvision.transforms.ToTensor"]
}
test_set = EsriChallengeDataset(test_cfg)

## DOTA
'''
train_cfg = {
    "root": "D:\\Documents\\AiTLAS\\aitlas\\ESRI-challenge\\data\\DOTA",
    "subset": "validation_split",
    "subsample_percentage": 0.01,
    "batch_size": 2,
    "shuffle": True,
    "num_workers": 0,
    "filter_null": True,
    "transforms": ["torchvision.transforms.ToTensor"]
}
train_set = DotaDataset(train_cfg)

test_cfg = {
    "root": "D:\\Documents\\AiTLAS\\aitlas\\ESRI-challenge\\data\\DOTA",
    "subset": "validation_split",
    "subsample_percentage": 0.01,
    "batch_size": 2,
    "shuffle": True,
    "num_workers": 0,
    "filter_null": True, 
    "transforms": ["torchvision.transforms.ToTensor"]
}
test_set = DotaDataset(train_cfg)
'''

# Model getter functions

def get_fastrcnn(num_classes = 3):
    model_cfg = {"num_classes": num_classes, "learning_rate": 0.001, "pretrained": True}

    # load a model pre-trained on COCO
    model = FastRCNN(model_cfg)

    model.prepare()

    return model

def get_retinanet(num_classes = 3):
    model_cfg = {"num_classes": num_classes, "learning_rate": 0.001, "pretrained": True}
    
    model = RetinaNet(model_cfg)

    model.prepare()
    
    return model


# Test the performance of FastRCNN
model = get_fastrcnn(num_classes = 3)

model.train_and_evaluate_model(
    train_dataset=train_set,
    val_dataset=test_set,
    epochs=3,
    model_directory = "./experiment/",
    run_id = "esri-fastrcnn",
    iterations_log = 30
)


# Test the performance of RetinaNet
'''
model = get_retinanet(num_classes = 3)

model.train_and_evaluate_model(
    train_dataset=train_set,
    val_dataset=test_set,
    epochs=20,
    model_directory = "./experiment/",
    run_id = "esri-retinanet",
    iterations_log = 40
)
'''
