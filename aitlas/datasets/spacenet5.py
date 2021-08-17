import os
from csv import reader

from aitlas.base import BaseDataset
from aitlas.datasets.schemas import SpaceNet5DatasetSchema
from ..utils import image_loader
from skimage import io


class SpaceNet5Dataset(BaseDataset):
    """
    SpaceNet5 dataset.

    Notes
    -----
        It is assumed that the 8-bit and speed mask conversion have been done, i.e. tasks 02 and 03.

        config.filenames points to the {train, tes, val}.csv output file by the SpaceNet5SplitTask
        config.image_root points to the image directory
        config.cont_mask_root points to the continuous mask directory
        config.mask_mc_root points to the multi-channel mask directory

    Attributes
    ----------
        filenames : list
            Contains the .tif filenames specified in the csv output file of the split task
    """
    schema = SpaceNet5DatasetSchema # set up the dataset schema

    def __init__(self, config):
        """
        Parameters
        ----------
            config : Config
                The configuration for the dataset.
        """
        BaseDataset.__init__(self, config)
        self.filenames = list()
        self.__load_filenames(config.filenames)

    def __getitem__(self, index):
        """
        Loads the dataset item at the specified index.
        Applies the transformations to the item before returning it.

        Parameters
        ----------
            index : int
                Specifying which item to return.

        Returns
        -------
            image, mask
                A pair of the source image, and its mask (i.e. segmentation ground-truth)
        """
        # Load the image
        img_path = os.path.join(self.config.image_root, self.filenames[index])
        image = image_loader(img_path)
        # Load the mask
        mc_mask_path = os.path.join(self.config.mc_mask_root, self.filenames[index])
        mask = io.imread(mc_mask_path)
        # Apply transformations
        image, mask = self.transform({"image": image, "mask": mask})
        return image, mask

    def get_image(self, index):
        """
        Helper method for the prediction task.

        Similar to __getitem__, this method only gets the image and not the mask,
        since at prediction time we do not have a segmentation ground-truth mask
        and thus we need to avoid the transformation that __getitem__ imposes on it.

        Parameters
        ----------
            index : int
                Specifying which image to return.

        Returns
        -------
            image
                The source image, transformed
        """
        img_path = os.path.join(self.config.image_root, self.filenames[index])
        image = image_loader(img_path)
        image, _ = self.transform({
            "image": image,
            "mask": image  # workaround to avoid the usage of masks
        })
        return image

    def __len__(self):
        return len(self.filenames)

    def get_filename(self, index):
        """
        Helper method for the prediction task.

        Returns the filename for the index-th image so that we can use it to save the image's prediction.

        Parameters
        ----------
            index : int
                Specifying which filename to return.

        Returns
        -------
            str
                the filename at the specified index
        """
        return self.filenames[index]

    def labels(self):
        """
        Hard-coded labels based on the csv burn-value output file from the task 03.
        TODO: These may not be needed and probably should be removed.
        """
        return {
            "0-10mph": 0,
            "11-20mph": 1,
            "21-30mph": 2,
            "31-40mph": 3,
            "41-50mph": 4,
            "51-60mph": 5,
            "61-65mph": 6
        }

    def __load_filenames(self, csv_path):
        """
        Reads the filenames from the csv file and loads them in the self.filenames list.

        Parameters
        ----------
            csv_path : str
                Path to the csv output file from the split task that contains the filenames of the items in the dataset.
        """
        with open(csv_path, "r") as reader_obj:
            csv_reader = reader(reader_obj)
            for row in csv_reader:
                self.filenames.append(os.path.basename(row[0]))
