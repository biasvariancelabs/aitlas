import numpy as np
from PIL import Image

from ..utils import image_loader
from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "Airplane",
    "Airport",
    "Artificial dense forest land",
    "Artificial sparse forest land",
    "Bare land",
    "Basketball court",
    "Blue structured factory building",
    "Building",
    "Construction site",
    "Cross river bridge",
    "Crossroads",
    "Dense tall building",
    "Dock",
    "Fish pond",
    "Footbridge",
    "Graff",
    "Grassland",
    "Low scattered building",
    "Lrregular farmland",
    "Medium density scattered building",
    "Medium density structured building",
    "Natural dense forest land",
    "Natural sparse forest land",
    "Oiltank",
    "Overpass",
    "Parking lot",
    "Plasticgreenhouse",
    "Playground",
    "Railway",
    "Red structured factory building",
    "Refinery",
    "Regular farmland",
    "Scattered blue roof factory building",
    "Scattered red roof factory building",
    "Sewage plant-type-one",
    "Sewage plant-type-two",
    "Ship",
    "Solar power station",
    "Sparse residential area",
    "Square",
    "Steelsmelter",
    "Storage land",
    "Tennis court",
    "Thermal power plant",
    "Vegetable plot",
    "Water",
]


class RSD46WHUDataset(MultiClassClassificationDataset):

    url = "https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU"
    labels = LABELS

    name = "RSD46-WHU dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image
        img = np.asarray(
            Image.fromarray(image_loader(self.data[index][0])).convert("RGB")
        )

        # apply transformations
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(self.data[index][1])
        return img, target
