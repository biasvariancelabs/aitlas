import logging

from ..base import BaseDataset, BaseModel, BaseTask
from ..utils import get_class, stringify
from .schemas import EvaluateTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class EvaluateTask(BaseTask):
    schema = EvaluateTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

    def run(self):
        """Evaluate the dataset against a given model"""

        # load the dataset
        dataset = self.create_dataset(self.config.dataset_config)

        dataset.prepare()

        # get metric classes
        metrics = []
        for metric in self.config.metrics:
            metrics.append(get_class(metric))

        calculated_metrics, y_true, y_pred, y_probs, loss = self.model.evaluate(
            dataset=dataset, model_path=self.config.model_path, metrics=metrics,
        )

        # get metric classes
        visualuzation_conf = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
            "loss": loss,
            "labels": dataset.labels(),
        }
        for vis in self.config.visualizations:
            viz_cls = get_class(vis)
            viz_cls(**visualuzation_conf).plot()

        logging.info(stringify(calculated_metrics))
