import csv
import logging
import os

from ..base import BaseDataset, BaseModel, BaseTask, Configurable
from ..utils import get_class, pil_loader, stringify
from ..visualizations import display_image_labels, display_image_segmentation
from .schemas import PredictLabelsTask, PredictTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class TestFolderDataset(BaseDataset):
    def __init__(self, root, labels, transform):
        BaseDataset.__init__(self, {})

        self.root = root
        self.labels = labels
        self.transform = transform

        self.data = []
        self.fnames = []

        dir = os.path.expanduser(self.root)
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                self.data.append(os.path.join(root, fname))
                self.fnames.append(fname)

    def __getitem__(self, index):
        img = self.data[index]
        return (
            self.transform(pil_loader(img)),
            0,
        )  # returning `0` because we have no target

    def __len__(self):
        return len(self.data)


class PredictTask(BaseTask):
    schema = PredictLabelsTask

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

        self.dir = self.config.dir
        self.output_path = self.config.output_path
        self.output_format = self.config.output_format

    def run(self):
        """Do something awesome here"""

        # load the dataset
        dataset = self.create_dataset(self.config.dataset_config)

        test_dataset = TestFolderDataset(self.dir, dataset.labels(), dataset.transform)

        # run predictions
        _, y_true, y_pred, y_prob, _ = self.model.evaluate(
            dataset=test_dataset, model_path=self.config.model_path, metrics=(),
        )

        if self.output_format == "plot":
            for i, image_path in enumerate(test_dataset.data):
                plot_path = os.path.join(
                    self.output_path, f"{test_dataset.fnames[i]}_plot.png"
                )
                # y_true, y_pred, y_prob, labels, file
                display_image_labels(
                    image_path,
                    y_true[i],
                    y_pred[i],
                    y_prob[i],
                    test_dataset.labels,
                    plot_path,
                )
        else:
            self.export_predictions_to_csv(
                self.output_path, test_dataset.fnames, y_prob, test_dataset.labels
            )

    def export_predictions_to_csv(self, file, fnames, probs, labels):
        with open(file, "w", newline="") as csvfile:
            fieldnames = ["image"] + labels
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()

            for i, fname in enumerate(fnames):
                obj = {label: probs[i][j] for j, label in enumerate(labels)}
                obj["image"] = fname

                writer.writerow(obj)


class PredictSegmentationTask(BaseTask):
    schema = PredictTaskSchema

    def run(self):
        """Do something awesome here"""

        # load the dataset
        dataset = self.create_dataset(self.config.dataset_config)

        test_dataset = TestFolderDataset(
            self.config.dir, dataset.labels(), dataset.transform
        )

        # run predictions
        _, y_true, y_pred, y_prob, _ = self.model.evaluate(
            dataset=test_dataset, model_path=self.config.model_path, metrics=(),
        )

        # plot predictions
        for i, image_path in enumerate(test_dataset.data):
            plot_path = os.path.join(
                self.config.output_path, f"{test_dataset.fnames[i]}_plot.png"
            )
            display_image_segmentation(
                image_path,
                y_true[i],
                y_pred[i],
                y_prob[i],
                test_dataset.labels,
                plot_path,
            )
