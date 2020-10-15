import logging

from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import SplitTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class PrepareTask(BaseTask):
    """If the prepare part (or a version of it) is extensive, you can run it as a separate task"""

    schema = SplitTaskSchema

    def __init__(self, model: BaseModel, dataset: BaseDataset, config):
        super().__init__(model, dataset, config)

    def run(self):
        self.dataset.prepare()
        logging.info("And that's it!")
