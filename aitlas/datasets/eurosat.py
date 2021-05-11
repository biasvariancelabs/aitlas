from .multiclass_classification import MultiClassClassificationDataset

LABELS = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture",
          "PermanentCrop", "Residential", "River", "SeaLake"]


class EurosatDataset(MultiClassClassificationDataset):

    labels = LABELS

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        MultiClassClassificationDataset.__init__(self, config)
