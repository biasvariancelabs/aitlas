import logging

import torch
from aitlas.base import BaseTask
from aitlas.tasks.schemas import EvaluateTaskSchema
from aitlas.utils import get_class, stringify
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SpaceNet5EvaluateTask(BaseTask):
    """Custom task for evaluating the model."""

    schema = EvaluateTaskSchema  # set up the schema for the task

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
                Which is to be evaluated.
            config : Config
                Specifying the details for this task.
        """
        super().__init__(model, config)

    def run(self):
        """Implements the main logic of this task."""
        # Load dataset and model
        dataset = self.create_dataset(self.config.dataset_config)
        self.model.load_model(self.config.model_path)
        self.model.eval()
        # Aggregation variables
        total_loss = 0.0
        n = len(dataset)
        # Get metric classes
        metrics = list()
        for metric in self.config.metrics:
            metrics.append(get_class(metric))
        # Run evaluation
        with torch.no_grad():
            for i, data in enumerate(
                tqdm(dataset.dataloader(), desc="[custom evaluation]")
            ):
                input_image, target_mask = data
                input_image = input_image.to(self.model.device)
                # Run prediction
                output_mask = self.model.predict(input_image)
                # Calculate and take into account batch loss
                # load_criterion() should return FocalLoss * .75 + DiceCoefficient * .25 for UNetResnet34
                batch_loss = self.model.load_criterion()(output_mask, target_mask)
                total_loss += batch_loss.item() * input_image.size(0)
                # Calculate some other metrics
                calculated_metrics = {}
                for metric_class in metrics:
                    metric = metric_class(device=self.model.device)  # load class
                    calculated_metrics.setdefault(metric.name, 0)
                    calculated_metrics[metric.name] += metric.calculate(
                        y_true=target_mask, y_pred=output_mask
                    )
            # Calculate total losses
            for metric_name, metric_loss in calculated_metrics.items():
                calculated_metrics[metric_name] = metric_loss / n
            total_loss = total_loss / n
        logging.info(f"Total loss: {total_loss}")
        logging.info(stringify(calculated_metrics))
