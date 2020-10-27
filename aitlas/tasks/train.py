from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import TrainAndEvaluateTaskSchema, TrainTaskSchema


class TrainTask(BaseTask):
    schema = TrainTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        """Do something awesome here"""
        dataset = self.create_dataset(self.config.dataset_config)

        dataset.prepare()

        self.model.train_model(
            train_dataset=dataset,
            epochs=self.config.epochs,
            model_directory=self.config.model_directory,
            save_epochs=self.config.save_epochs,
            resume_model=self.config.resume_model,
            run_id=self.id,
            iterations_log=self.config.iterations_log,
            metrics=self.model.metrics(),
        )


class TrainAndEvaluateTask(BaseTask):
    schema = TrainAndEvaluateTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        """Do something awesome here"""
        train_dataset = self.create_dataset(self.config.train_dataset_config)
        val_dataset = self.create_dataset(self.config.val_dataset_config)

        self.model.train_and_evaluate_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=self.config.epochs,
            model_directory=self.config.model_directory,
            save_epochs=self.config.save_epochs,
            resume_model=self.config.resume_model,
            run_id=self.id,
            iterations_log=self.config.iterations_log,
            metrics=self.model.metrics(),
        )
