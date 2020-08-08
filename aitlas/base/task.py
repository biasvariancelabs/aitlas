from .config import Configurable


class BaseTask(Configurable):
    def __init__(self, model, dataset, config):
        super().__init__(config)

        self.model = model
        self.dataset = dataset

    def run(self):
        """Runs the task."""
        raise NotImplementedError
