from .config import Configurable


class BaseTask(Configurable):
    def __init__(self, model, config):
        super().__init__(config)

        self.model = model

    def run(self):
        """Runs the task."""
        raise NotImplementedError
