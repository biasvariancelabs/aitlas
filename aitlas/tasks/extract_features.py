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

        self.dir = self.config.dir
        self.output_dir = self.config.output_dir
        self.transforms = self.config.transforms

    def run(self):
        """Do something awesome here"""

        # set model in eval model
        self.model.eval()

        # run through the directory
        with torch.no_grad():
            dir = os.path.expanduser(self.dir)
            for root, _, fnames in sorted(os.walk(dir)):
                for fname in sorted(fnames):
                    full_path = os.path.join(root, fname)
                    img = image_loader(full_path)
                    feats = self.model(
                        load_transforms(self.transforms, self.config)(img).unsqueeze(0)
                    )

                    np.savetxt(
                        os.path.join(self.output_dir, f"{fname}.feat"), feats.numpy()
                    )
