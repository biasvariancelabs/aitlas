import os

from ..utils import image_loader
from .schemas import ClusteringDatasetSchema
from .clustering import ClusteringDataset 

"""
These datasets were used in the AI4QC project (Artificial Intelligence for Quality Control), in the context of the 
detection of new anomalies through unsupervised learning (unlabeled data). It consists of 6344 Sentinel-1 images 
(quick-looks in png format), and 6452 Sentinel-2 images (true color images in jpg format).
"""


class S1NewAnomaliesDataset(ClusteringDataset):

    url = "https://zenodo.org/records/13789234"
    name = "S1 New Anomalies dataset"
    schema = ClusteringDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)

        # load the data
        self.data_dir = self.config.data_dir
        self.images = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int

        """
        # load image
        image = image_loader(self.images[index])
        images = self.apply_transformations(image)
        return images 

    def load_dataset(self, data_dir, csv_file=None):
        ids = os.listdir(data_dir)
        self.images = [os.path.join(data_dir, image_id) for image_id in ids]
        return self.images
    
    
class S2NewAnomaliesDataset(ClusteringDataset):

    url = "https://zenodo.org/records/13790109"
    name = "S2 New Anomalies dataset"
    schema = ClusteringDatasetSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)

        # load the data
        self.data_dir = self.config.data_dir
        self.images = self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int

        """
        # load image
        image = image_loader(self.images[index])
        images = self.apply_transformations(image)
        return images 

    def load_dataset(self, data_dir, csv_file=None):
        ids = os.listdir(data_dir)
        self.images = [os.path.join(data_dir, image_id) for image_id in ids]
        return self.images