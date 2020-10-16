import logging

from ..base import BaseDataset, BaseModel, BaseTask
from ..utils import get_class, stringify
from .schemas import EvaluateTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EvaluateTask(BaseTask):
    schema = EvaluateTaskSchema

    def __init__(self, model: BaseModel, dataset: BaseDataset, config):
        super().__init__(model, dataset, config)

    def run(self):
        """Do something awesome here"""

        # prepare the dataset
        self.dataset.prepare()

        # get metric classes
        metrics = []
        for metric in self.config.metrics:
            metrics.append(get_class(metric))

        calculated_metrics, y_true, y_pred, loss = self.model.evaluate(
            dataset=self.dataset, model_path=self.config.model_path, metrics=metrics,
        )

        # get metric classes
        visualuzation_conf = {
            "y_true": y_true,
            "y_pred": y_pred,
            "loss": loss,
            "dataset": self.dataset,
        }
        for vis in self.config.visualizations:
            viz_cls = get_class(vis)
            viz_cls(**visualuzation_conf).plot()

        logging.info(stringify(calculated_metrics))
