import os

import torch
from aitlas.base import BaseTask
from aitlas.tasks.schemas import SplitTaskSchema
from torch.utils.data import random_split


class SpaceNet5SplitTask(BaseTask):
    """
    Task for splitting the SpaceNet5 dataset into train, test and validation sets.

    Attributes
    ----------
        filenames : list
            Contains the .tif filenames of the config.root directory.
    """

    schema = SplitTaskSchema

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
            config : Config

        Raises
        ------
            ValueError
                If the sum of the ratios for the splits is not equal to 100.
        """
        super().__init__(model, config)
        if not self.is_valid_split():
            raise ValueError("the sum of ratios for the splits must be equal to 100")
        self.filenames = [
            f for f in os.listdir(path=self.config.root) if f.endswith(".tif")
        ]

    def is_valid_split(self):
        """
        Checks if the specified split is valid.

        Returns
        -------
            bool
                Specifying whether the sum of ratios is equal to 100%.
        """
        ratio_sum = self.config.split.train.ratio + self.config.split.test.ratio
        if self.has_validation_dataset():
            ratio_sum += self.config.split.val.ratio
        return ratio_sum == 100

    def has_validation_dataset(self):
        """
        Checks whether the split contains a validation set.

        Returns
        ------
            bool
                Specifying the presence of a split.val attribute in the configuration
        """
        return self.config.split.val

    def save_split(self, split_indices, destination_file):
        """
        Saves the split by writing all the filenames indexed by split_indices as rows in the destination_file.

        Parameters
        ----------
            split_indices : list
                A list of indices from the filenames list which belong to this split.
            destination_file : str
                A filepath specifying the destination for writing/saving the filenames for this split.
        """
        if not os.path.exists(os.path.dirname(destination_file)):
            os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        with open(destination_file, "w") as f:
            for inx in split_indices:
                f.write(
                    "{}\n".format(os.path.join(self.config.root, self.filenames[inx]))
                )
        f.close()

    def run(self):
        """
        Main method that runs the logic of the split task.

        First, it calculates the sizes of the total dataset and the train, test and validation sets.
        Then, based on their ratio it randomly splits up the index space of the dataset,
        generating arrays of the appropriate size. Lastly, each index-split alongside
        with the destination file are passed to the save_split method to save the split to disk.

        Returns
        -------
            None
        """
        dataset_size = len(self.filenames)
        train_dataset_size = int(dataset_size * self.config.split.train.ratio / 100)
        test_dataset_size = int(dataset_size * self.config.split.test.ratio / 100)
        dataset_lengths = [train_dataset_size, test_dataset_size]
        if self.has_validation_dataset():
            val_dataset_size = int(dataset_size * self.config.split.val.ratio / 100)
            dataset_lengths.append(val_dataset_size)
        # Fixes roundup cases by appending the leftover to the training set
        dataset_lengths[0] += dataset_size - sum(dataset_lengths)
        split_indices = random_split(
            dataset=range(dataset_size),
            lengths=dataset_lengths,
            generator=torch.Generator().manual_seed(42),
        )
        # Save train and test split
        self.save_split(
            split_indices=split_indices[0],
            destination_file=self.config.split.train.file,
        )
        self.save_split(
            split_indices=split_indices[1], destination_file=self.config.split.test.file
        )
        # Save validation split
        if self.has_validation_dataset():
            self.save_split(
                split_indices=split_indices[2],
                destination_file=self.config.split.val.file,
            )
