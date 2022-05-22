import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..base import BaseModel, BaseTask
from .schemas import VisualizeTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class VisualizeTask(BaseTask):
    schema = VisualizeTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def get_distribution(self, split, split_type):
        if split:
            dataset = self.create_dataset(split)

            df = dataset.data_distribution_table()
            df["Split"] = split_type

            return df, dataset.get_name()

        return None, None

    def get_distribution_fig(self):
        # load the datasets and distributions
        print(self.config.split.train)
        train, train_name = self.get_distribution(self.config.split.train, "Train")
        val, val_name = self.get_distribution(self.config.split.val, "Val")
        test, test_name = self.get_distribution(self.config.split.test, "Test")

        name = (
            train_name
            if train_name
            else test_name
            if test_name
            else val_name
            if val_name
            else ""
        )

        label_count = pd.concat([train, val, test])
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(y="Label", x="Count", hue="Split", data=label_count)
        ax.set_title("Image distribution for {}".format(name), pad=20, fontsize=18)
        return fig

    def run(self):
        """Visualize the distributoin of the dataset"""

        logging.info("Loading config...")

        fig = self.get_distribution_fig()

        logging.info(f"Saving plot to {self.config.output_file}")

        fig.savefig(self.config.output_file, format="png")

        logging.info("Done!")
