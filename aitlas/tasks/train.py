import numpy as np

from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import OptimizeTaskSchema, TrainAndEvaluateTaskSchema, TrainTaskSchema


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
            metrics=self.model.metrics,
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
            metrics=self.model.metrics,
        )


def generate_parameters_for_range(range, steps):
    return np.arange(range[0], range[1], (range[1] - range[0]) / steps)


def generate_parameters(method, parameters, steps):
    """Generate parameters to search"""
    names = [parameter.name for parameter in parameters]

    values = []
    for parameter in parameters:
        ranges = generate_parameters_for_range(parameter.range, steps)
        values.append(ranges)

    total = np.array(np.meshgrid(*values)).T.reshape(-1, len(parameters))

    for row in total:
        parameter_set = []
        for i, name in enumerate(names):
            parameter_set.append({"name": name, "value": row[i]})

        yield parameter_set


class OptimizeTask(BaseTask):
    """
    Optimize certain parameters for the models
    """

    schema = OptimizeTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        """Do something awesome here"""

        train_dataset = self.create_dataset(self.config.train_dataset_config)
        val_dataset = self.create_dataset(self.config.val_dataset_config)

        print("done")
        print(self.config.parameters)

        parameters = generate_parameters(
            self.config.method, self.config.parameters, self.config.steps
        )

        best_parameters = None
        best_loss = None
        loss = 0

        for parameter_set in parameters:

            print(parameter_set)
            for parameter in parameter_set:
                setattr(self.model.config, parameter["name"], parameter["value"])

            loss = self.model.train_and_evaluate_model(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=self.config.epochs,
                model_directory=self.config.model_directory,
                save_epochs=self.config.epochs,
                run_id=self.id,
                iterations_log=100,
                metrics=self.model.metrics,
            )

            if not best_loss or loss < best_loss:
                best_loss = loss
                best_parameters = parameter_set

        print(best_parameters)
