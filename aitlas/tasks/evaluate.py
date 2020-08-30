import logging

from ..base import BaseDataset, BaseModel, BaseTask
from ..utils import stringify
from .schemas import EvaluateTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EvaluateTask(BaseTask):
    schema = EvaluateTaskSchema

    def __init__(self, model: BaseModel, dataset: BaseDataset, config):
        super().__init__(model, dataset, config)

    def run(self):
        """Do something awesome here"""
        result = self.model.evaluate(
            dataset=self.dataset,
            model_path=self.config.model_path,
            metrics=self.config.metrics,
        )

        logging.info(stringify(result))
