import logging
import os
from shutil import copyfile

import numpy as np

from ..base import BaseDataset, BaseModel, BaseTask
from .schemas import OptimizeTaskSchema, TrainAndEvaluateTaskSchema, TrainTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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


def generate_parameters_for_range(method, parameter):
    if method == "grid":
        return np.arange(
            parameter.min,
            parameter.max,
            (parameter.max - parameter.min) / parameter.steps,
        )
    elif method == "random":
        return np.random.uniform(
            low=parameter.min, high=parameter.max, size=(parameter.steps,)
        )
    else:
        raise ValueError("Incorrect parameter search method!")


def generate_parameters(method, parameters):
    """Generate parameters to search"""
    names = [parameter.name for parameter in parameters]

    values = []
    for parameter in parameters:
        ranges = generate_parameters_for_range(method, parameter)
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
        logging.info(f"Searching parameters")
        train_dataset = self.create_dataset(self.config.train_dataset_config)
        val_dataset = self.create_dataset(self.config.val_dataset_config)

        parameters = generate_parameters(self.config.method, self.config.parameters)

        best_parameters = None
        best_run_id = None
        best_model_output_directory = os.path.join(self.config.model_directory, "best")
        best_loss = None
        loss = 0

        for i, parameter_set in enumerate(parameters):
            logging.info(f"Testing {i} for parameters: {parameter_set}")
            run_id = f"{self.id}-{i}"

            for parameter in parameter_set:
                setattr(self.model.config, parameter["name"], parameter["value"])

            loss = self.model.train_and_evaluate_model(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=self.config.epochs,
                model_directory=self.config.model_directory,
                save_epochs=self.config.epochs,
                run_id=run_id,
                iterations_log=100,
                metrics=self.model.metrics,
            )

            if not best_loss or loss < best_loss:
                best_loss = loss
                best_parameters = parameter_set
                best_run_id = run_id

        logging.info(f"Best parameters: {best_parameters}")

        if not os.path.isdir(best_model_output_directory):
            os.makedirs(best_model_output_directory)

        checkpoint = sorted(
            filter(
                lambda x: "checkpoint" in x,
                os.listdir(os.path.join(self.config.model_directory, best_run_id)),
            ),
            reverse=True,
        )[0]

        copyfile(
            checkpoint, os.path.join(best_model_output_directory, "checkpoint.pth.tar")
        )
        logging.info(
            f"Best models saved at: {os.path.join(best_model_output_directory, 'checkpoint.pth.tar')}"
        )
