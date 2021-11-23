import csv
import logging
import os

import numpy as np
import torch

from ..base import BaseModel, BaseTask, load_transforms
from ..utils import image_loader
from .schemas import ExtractFeaturesTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ExtractFeaturesTask(BaseTask):
    schema = ExtractFeaturesTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

        self.data_dir = self.config.data_dir
        self.output_dir = self.config.output_dir
        self.transforms = self.config.transforms

    def run(self):
        """Do something awesome here"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # set the model to extract feature only
        self.model.extract_features()

        # load the model from disk if specified
        if self.config.model_path:
            self.model.load_model(self.config.model_path)

        # allocate device
        self.model.allocate_device()

        # set model in eval model
        self.model.eval()

        # run through the directory
        with torch.no_grad():
            data_dir = os.path.expanduser(self.data_dir)
            for root, _, fnames in sorted(os.walk(data_dir)):
                for fname in sorted(fnames):
                    full_path = os.path.join(root, fname)
                    img = image_loader(full_path)
                    input = load_transforms(self.transforms, self.config)(img).to(
                        device
                    )
                    feats = self.model(input.unsqueeze(0))

                    # move the features to cpu if not there
                    if device != "cpu":
                        feats = feats.cpu()

                    np.savetxt(
                        os.path.join(self.output_dir, f"{fname}.feat"), feats.numpy().flatten(),
                    )

        logging.info(f"And that's it! The features are in {self.output_dir}")
