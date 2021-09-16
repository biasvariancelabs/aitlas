from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


class EurosatDataset(MultiClassClassificationDataset):

    url = "https://github.com/phelber/EuroSAT"

    labels = LABELS
    name = "EuroSAT dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
