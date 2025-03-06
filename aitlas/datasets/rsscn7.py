from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "farm_land",
    "forest",
    "grass_land",
    "industrial_region",
    "parking_lot",
    "residential_region",
    "river_lake",
]


class RSSCN7Dataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/nifulislam/rsscn7-dataset"

    labels = LABELS
    name = "RSSCN7 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
