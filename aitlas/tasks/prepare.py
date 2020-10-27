import logging

from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import PrepareTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class PrepareTask(BaseTask):
    """If the prepare part (or a version of it) is extensive, you can run it as a separate task"""

    schema = PrepareTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        """Do some offline preparation"""

        dataset = self.create_dataset(self.config.dataset_config)
        dataset.prepare()

        logging.info("And that's it!")
