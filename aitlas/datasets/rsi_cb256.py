from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "airplane",
    "airport_runway",
    "artificial_grassland",
    "avenue",
    "bare_land",
    "bridge",
    "city_building",
    "coastline",
    "container",
    "crossroads",
    "dam",
    "desert",
    "dry_farm",
    "forest",
    "green_farmland",
    "highway",
    "hirst",
    "lakeshore",
    "mangrove",
    "marina",
    "mountain",
    "parkinglot",
    "pipeline",
    "residents",
    "river",
    "river_protection_forest",
    "sandbeach",
    "sapling",
    "sea",
    "shrubwood",
    "snow_mountain",
    "sparse_forest",
    "storage_room",
    "stream",
    "town",
]


class RSICB256Dataset(MultiClassClassificationDataset):

    url = "https://github.com/lehaifeng/RSI-CB"
    labels = LABELS

    name = "RSI-CB256 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
