from .multi_class_csv import MultiClassCsvDataset


CLASSES_TO_IDX = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}


class EurosatDataset(MultiClassCsvDataset):

    classes_to_idx = CLASSES_TO_IDX

    def __init__(self, config):
        # now call the constuctor to validate the schema and load the data
        MultiClassCsvDataset.__init__(self, config)
