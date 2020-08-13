from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import EvaluateTaskSchema


class EvaluateTask(BaseTask):
    schema = EvaluateTaskSchema

    def __init__(self, model: BaseModel, dataset: BaseDataset, config):
        super().__init__(model, dataset, config)

    def run(self):
        """Do something awesome here"""
        self.model.evaluate(
            dataset=self.testloader(), model_path=self.config.model_path,
        )
