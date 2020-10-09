from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import TrainTaskSchema


class TrainTask(BaseTask):
    schema = TrainTaskSchema

    def __init__(self, model: BaseModel, dataset: BaseDataset, config):
        super().__init__(model, dataset, config)

    def run(self):
        """Do something awesome here"""
        self.model.fit(
            dataset=self.dataset,
            epochs=self.config.epochs,
            model_directory=self.config.model_directory,
            save_epochs=self.config.save_epochs,
            resume_model=self.config.resume_model,
            run_id=self.id,
            iterations_log=self.config.iterations_log,
        )
