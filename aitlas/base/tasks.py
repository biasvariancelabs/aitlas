from ..utils import current_ts, get_class
from .config import Configurable
from .datasets import BaseDataset


class BaseTask(Configurable):
    def __init__(self, model, config):
        super().__init__(config)

        self.model = model

        # generate a task ID if not specified
        id = self.config.id
        if not id:
            id = str(self.generate_task_id())
        self.id = id

    @staticmethod
    def create_dataset(dataset_config):
        """ Builds the input dataset using the provided configuration. """

        classname, config = dataset_config.classname, dataset_config.config

        cls = get_class(classname)
        if not issubclass(cls, BaseDataset):
            raise ValueError("Data input class does not inherit from BaseInput.")

        dataset = cls(config)

        return dataset

    def generate_task_id(self):
        """Generates a task ID"""
        return current_ts()

    def run(self):
        """Runs the task."""
        raise NotImplementedError


