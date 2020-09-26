from ..utils import current_ts
from .config import Configurable


class BaseTask(Configurable):
    def __init__(self, model, dataset, config):
        super().__init__(config)

        self.model = model
        self.dataset = dataset

        # generate a task ID if not specified
        id = self.config.id
        if not id:
            id = str(self.generate_task_id())
        self.id = id

    def generate_task_id(self):
        """Generates a task ID"""
        return current_ts()

    def run(self):
        """Runs the task."""
        raise NotImplementedError
