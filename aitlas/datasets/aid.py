from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "Airport",
    "BareLand",
    "BaseballField",
    "Beach",
    "Bridge",
    "Center",
    "Church",
    "Commercial",
    "DenseResidential",
    "Desert",
    "Farmland",
    "Forest",
    "Industrial",
    "Meadow",
    "MediumResidential",
    "Mountain",
    "Park",
    "Parking",
    "Playground",
    "Pond",
    "Port",
    "RailwayStation",
    "Resort",
    "River",
    "School",
    "SparseResidential",
    "Square",
    "Stadium",
    "StorageTanks",
    "Viaduct",
]


class AIDDataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets"

    labels = LABELS
    name = "AID dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        MultiClassClassificationDataset.__init__(self, config)
