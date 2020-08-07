from ..base import BaseTask
from .schemas import TrainTaskSchema


class TrainTask(BaseTask):
    schema = TrainTaskSchema

    def __init__(self, model, config):
        super().__init__(model, config)

    def run(self):
        """Do something awesome here"""
        pass
