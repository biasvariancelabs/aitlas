import logging

from ..base import BaseModel, BaseTask
from ..utils import stringify
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

        loss = self.model.evaluate(dataset=dataset, model_path=self.config.model_path)

        # log the metrics
        logging.info(stringify(self.model.running_metrics.get_scores(self.model.metrics)))

        # generate a report
        self.model.report(dataset.get_labels(), dataset.get_name(), self.model.running_metrics, id=self.id)
