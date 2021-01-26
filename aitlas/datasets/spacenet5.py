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
        It is assumed that the 8-bit and speed mask conversion have been done.

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
                A pair of the source image, and its mask (i.e. segmentation ground truth)
        """
        # Load the image
        img_path = os.path.join(self.config.image_root, self.filenames[index])
        image = image_loader(img_path)
        # Load the mask
        mc_mask_path = os.path.join(self.config.mc_mask_root, self.filenames[index])
        mask = io.imread(mc_mask_path)
        # Apply transformations
        # TODO: Use self.load_transforms() to compose transformations instead of CompositeTransformations() ?
        image, mask = self.transform({"image": image, "mask": mask})
        return image, mask

    def __len__(self):
        return len(self.filenames)

    def get_filename(self, index):
        return self.filenames[index]

    def labels(self):
        raise NotImplementedError

    def __load_filenames(self, csv_path):
        """
        Reads the filenames from the csv file and saves them in the self.filenames list.

        Parameters
        ----------
            csv_path : str
                Path to the csv output file of the split task which contains the filenames for the items in this dataset.
        """
        with open(csv_path, "r") as reader_obj:
            csv_reader = reader(reader_obj)
            for row in csv_reader:
                self.filenames.append(os.path.basename(row[0]))
