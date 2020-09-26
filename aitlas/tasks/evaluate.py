import logging

from ..base import BaseDataset, BaseModel, BaseTask
from ..utils import get_class, stringify
from ..visualizations import ConfusionMatrix
from .schemas import EvaluateTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EvaluateTask(BaseTask):
    schema = EvaluateTaskSchema

    def __init__(self, model: BaseModel, dataset: BaseDataset, config):
        super().__init__(model, dataset, config)

    def create_visualizations(self):
        pass

    def run(self):
        """Do something awesome here"""

        # get metric classes
        metrics = []
        for metric in self.config.metrics:
            metrics.append(get_class(metric))

        calculated_metrics, y_true, y_pred, _ = self.model.evaluate(
            dataset=self.dataset, model_path=self.config.model_path, metrics=metrics,
        )

        viz = ConfusionMatrix(
            dataset=self.dataset,
            y_true=y_true,
            y_pred=y_pred,
            file="./examples/confusion.png",
        )
        fig = viz.plot()

        logging.info(stringify(calculated_metrics))
