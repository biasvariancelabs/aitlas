from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "airport",
    "bare-land",
    "beach",
    "bridge",
    "commercial",
    "desert",
    "farmland",
    "forest",
    "golf-course",
    "highway",
    "industrial",
    "meadow",
    "mountain",
    "overpass",
    "park",
    "parking",
    "playground",
    "port",
    "railway",
    "railway-station",
    "residential",
    "river",
    "runway",
    "stadium",
    "storage-tank",
]


class CLRSDataset(MultiClassClassificationDataset):

    url = "https://github.com/lehaifeng/CLRS"
    labels = LABELS
    name = "CLRS dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
