# datasets
from aitlas.datasets.esri_challenge import EsriChallengeDataset
from aitlas.datasets.dota import DotaDataset

# models
from aitlas.models.fastrcnn_detector import FastRCNN
from aitlas.models.retinanet import RetinaNet

# vizualization
from matplotlib import pyplot as plt
from matplotlib import patches
import matplotlib as mpl
import numpy as np

import argparse
import torch

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_device", help="sepcify which CUDA device should be used", type = int)
    parser.add_argument("--dataset", help = "which dataset should be used for training and testing (DOTA/ESRI)", type = str)
    parser.add_argument("--model", help = "which model should be used for training and testing (FastRCNN/RetinaNet)", type = str)
    parser.add_argument("--epochs", help = "How many epochs should the traning last", type = int)

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_device):

        if args.dataset == "ESRI":
            train_cfg = {
                "root": "/home/dkocev/data/ESRI",
                "subset": "train",
                "subsample_percentage": 1,
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0,
                "transforms": ["torchvision.transforms.ToTensor"]
            }
            train_set = EsriChallengeDataset(train_cfg)

            test_cfg = {
                "root": "/home/dkocev/data/ESRI",
                "subset": "test",
                "subsample_percentage": 1,
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 0,
                "transforms": ["torchvision.transforms.ToTensor"]
            }
            test_set = EsriChallengeDataset(test_cfg)
            num_classes = 3

        elif args.dataset == "DOTA":
            train_cfg = {
                "root": "/home/dkocev/data/DOTA",
                "subset": "trainsplit",
                "subsample_percentage": 0.01,
                "batch_size": 32,
                "shuffle": False,
                "num_workers": 0,
                "filter_null": True,
                "transforms": ["torchvision.transforms.ToTensor"]
            }
            train_set = DotaDataset(train_cfg)

            test_cfg = {
                "root": "/home/dkocev/data/DOTA",
                "subset": "validationsplit",
                "subsample_percentage": 0.01,
                "batch_size": 32,
                "shuffle": False,
                "num_workers": 0,
                "filter_null": True, 
                "transforms": ["torchvision.transforms.ToTensor"]
            }
            test_set = DotaDataset(test_cfg)
            num_classes = 16
        else:
            print ("Invalid dataset selection")
            exit(-1)

        if args.model == 'FastRCNN':
            # Test the performance of FastRCNN
            model = get_fastrcnn(num_classes = num_classes)

            model.train_model(
                train_dataset=train_set,
                val_dataset=test_set,
                epochs=args.epochs,
                model_directory = "./experiment/",
                run_id = "{}-fastrcnn".format(args.dataset),
                iterations_log = 20
            )
        elif args.model == 'RetinaNet':
            # Test the performance of RetinaNet
            model = get_retinanet(num_classes = num_classes)

            model.train_model(
                train_dataset=train_set,
                val_dataset=test_set,
                epochs=args.epochs,
                model_directory = "./experiment/",
                run_id = "{}-retinanet".format(args.dataset),
                iterations_log = 20
            )
        else:
            print ("Invalid model selection")
            exit(-1)