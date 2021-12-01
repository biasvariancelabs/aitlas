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

    url = "https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxxaW56b3VjbnxneDo1MDYzYWMxOWIwMjRiMWFi"

    labels = LABELS
    name = "RSSCN7 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
