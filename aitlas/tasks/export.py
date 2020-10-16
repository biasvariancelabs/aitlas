import logging

from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import ExportSplitTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ExportSplitsTask(BaseTask):
    """Exports the internal split indices to dataset IDs. This might be useful for someone to use the splits elsewhere"""

    schema = ExportSplitTaskSchema

    def __init__(self, model: BaseModel, dataset: BaseDataset, config):
        super().__init__(model, dataset, config)

    def run(self):
        self.dataset.prepare()

        data = self.get_data(self.dataset.train_indices)
        self.export_split(self.config.train_export, data)

        data = self.get_data(self.dataset.test_indices)
        self.export_split(self.config.test_export, data)

        if self.config.val_export and self.dataset.val_indices:
            data = self.get_data(self.dataset.val_indices)
            self.export_split(self.config.val_export, data)

        logging.info("And that's it!")

    def export_split(self, file, data):
        with open(file, "w") as f:
            f.write("index;name\n")
            for ind, name in data:
                f.write(f"{ind};{name}\n")

    def get_data(self, indices):
        return [(index, self.dataset.get_item_name(index)) for index in indices]
