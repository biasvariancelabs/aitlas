import csv
import logging
import os

from ..base import BaseDataset, BaseModel, BaseTask, Configurable
from ..utils import get_class, image_loader, stringify
from ..visualizations import (
    display_eopatch_predictions,
    display_image_labels,
    display_image_segmentation,
    save_predicted_masks,
)
from .schemas import PredictTaskSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ImageFolderDataset(BaseDataset):
    def __init__(self, data_dir, labels, transforms, batch_size):
        BaseDataset.__init__(self, {})

        self.data_dir = data_dir
        self.labels = labels
        self.transform = self.load_transforms(transforms)
        self.shuffle = False
        self.batch_size = batch_size

        self.data = []
        self.fnames = []

        data_dir = os.path.expanduser(self.data_dir)
        for root, _, fnames in sorted(os.walk(data_dir)):
            for fname in sorted(fnames):
                self.data.append(os.path.join(root, fname))
                self.fnames.append(fname)

    def __getitem__(self, index):
        img = self.data[index]
        return (
            self.transform(image_loader(img)),
            0,
        )  # returning `0` because we have no target

    def __len__(self):
        return len(self.data)


class PredictTask(BaseTask):
    schema = PredictTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

        self.data_dir = self.config.data_dir
        self.output_dir = self.config.output_dir
        self.output_file = self.config.output_file
        self.output_format = self.config.output_format

    def run(self):
        """Do something awesome here"""

        # load the configs
        if self.config.dataset_config:
            dataset = self.create_dataset(self.config.dataset_config)
            labels = dataset.get_labels()
            transforms = dataset.config.transforms
            batch_size = dataset.config.batch_size
        else:
            labels = self.config.labels
            transforms = self.config.transforms
            batch_size = self.config.batch_size

        test_dataset = ImageFolderDataset(
            self.data_dir, labels, transforms, batch_size,
        )

        # load the model
        self.model.load_model(self.config.model_path)

        # run predictions
        y_true, y_pred, y_prob = self.model.predict(dataset=test_dataset,)

        if self.output_format == "plot":
            for i, image_path in enumerate(test_dataset.data):
                plot_path = os.path.join(
                    self.output_dir, f"{test_dataset.fnames[i]}_plot.png"
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
                self.output_file, test_dataset.fnames, y_prob, test_dataset.labels
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

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

        self.output_format = self.config.output_format

    def run(self):
        """Do something awesome here"""

        # load the configs
        if self.config.dataset_config:
            dataset = self.create_dataset(self.config.dataset_config)
            labels = dataset.get_labels()
            transforms = dataset.config.transforms
            batch_size = dataset.config.batch_size
        else:
            labels = self.config.labels
            transforms = self.config.transforms
            batch_size = self.config.batch_size

        test_dataset = ImageFolderDataset(
            self.config.data_dir, labels, transforms, batch_size,
        )

        # load the model
        self.model.load_model(self.config.model_path)

        # run predictions
        y_true, y_pred, y_prob = self.model.predict(dataset=test_dataset,)

        if self.output_format == "plot":
            # plot predictions
            for i, image_path in enumerate(test_dataset.data):
                plot_path = os.path.join(
                    self.config.output_dir, f"{test_dataset.fnames[i]}_plot.png"
                )
                display_image_segmentation(
                    image_path,
                    y_true[i],
                    y_pred[i],
                    y_prob[i],
                    test_dataset.labels,
                    plot_path,
                )
        else:
            # save raw masks
            for i, image_path in enumerate(test_dataset.data):
                base_filepath_name = os.path.join(
                    self.config.output_dir, os.path.splitext(test_dataset.fnames[i])[0]
                )
                save_predicted_masks(
                    y_pred[i], test_dataset.labels, base_filepath_name,
                )


class PredictEOPatchTask(BaseTask):
    schema = PredictTaskSchema

    def __init__(self, model: BaseModel, config):
        super().__init__(model, config)

        self.dir = self.config.dir
        self.output_path = self.config.output_path  # use this
        self.output_format = self.config.output_format

    def run(self):
        """Do something awesome here"""

        # load the configs
        if self.config.dataset_config:
            dataset = self.create_dataset(self.config.dataset_config)
            labels = dataset.get_labels()
            transforms = dataset.config.transforms
        else:
            raise ValueError("Please provide a test dataset config.")

        test_dataset = dataset
        # load the model
        self.model.load_model(self.config.model_path)

        # run predictions
        y_true, y_pred, y_prob = self.model.predict(dataset=test_dataset,)

        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        if self.output_format == "plot":
            # assume all eopatches in the test dataset are in the same "eopatches" folder
            eopatches_path = os.path.join(test_dataset.root, "eopatches")
            test_index = test_dataset.index

            # this for should be in a separate function
            for f in os.scandir(eopatches_path):  # TODO: the dataset should return this
                if f.is_dir():
                    patch = f.name

                    fig = display_eopatch_predictions(
                        eopatches_path,
                        patch,
                        y_pred,
                        test_index,
                        y_true,
                        test_dataset.mapping,
                    )

                    fig.savefig(
                        f"{self.output_path}{os.sep}{patch}__visual_predictions.png",
                        dpi=300,
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
