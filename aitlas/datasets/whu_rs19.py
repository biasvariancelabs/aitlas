from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "Airport",
    "Beach",
    "Bridge",
    "Commercial",
    "Desert",
    "Farmland",
    "footballField",
    "Forest",
    "Industrial",
    "Meadow",
    "Mountain",
    "Park",
    "Parking",
    "Pond",
    "Port",
    "railwayStation",
    "Residential",
    "River",
    "Viaduct",
]


class WHURS19Dataset(MultiClassClassificationDataset):

    url = "https://github.com/CAPTAIN-WHU/BED4RS"
    labels = LABELS
    name = "WHU-RS19 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
