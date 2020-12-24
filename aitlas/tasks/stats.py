import logging

import numpy as np
from tqdm import tqdm

from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import PrepareTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class StatsTask(BaseTask):
    """Get some basic stats for a dataset """

    schema = PrepareTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        """Do some offline preparation"""

        dataset = self.create_dataset(self.config.dataset_config)

        # data set length
        total = len(dataset)
        logging.info(f"Number of items: {total}")

        # distributions per labels
        labels = dataset.labels()
        counts = {dataset.classes_to_idx[label]: 0 for label in labels}

        for i, data in enumerate(tqdm(dataset.dataloader(), desc="counting")):
            item, target = data

            for minitarget in target:
                minitarget = minitarget.item()
                if isinstance(minitarget, np.ndarray):
                    for label in minitarget:
                        counts[label] += 1
                else:
                    counts[minitarget] += 1

        logging.info(f"Items per label")
        logging.info(f"-----------------------------------")
        for label in labels:
            count = counts[dataset.classes_to_idx[label]]
            logging.info(f"{label}: {count} ~ {round(count/total * 100)}%")
