import logging

from ..base import BaseDataset, BaseModel, BaseTask
from ..utils import get_class, stringify
from .schemas import EvaluateTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EvaluateTask(BaseTask):
    schema = EvaluateTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        """Evaluate the dataset against a given model"""

        # load the dataset
        dataset = self.create_dataset(self.config.dataset_config)

        # get metric classes
        metrics = []
        for metric in self.config.metrics:
            metrics.append(get_class(metric))

        calculated_metrics, y_true, y_pred, y_prob, loss = self.model.evaluate(
            dataset=dataset, model_path=self.config.model_path, metrics=metrics,
        )

        # log the metrics
        logging.info(stringify(calculated_metrics))

        # generate a report
        self.model.report(y_true, y_pred, y_prob, dataset.labels(), id=self.id)
