from .multilabel_classification import MultiLabelClassificationDataset


LABELS = [
    "airplane",
    "bare-soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile-home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "tanks",
    "trees",
    "water",
]


class UcMercedMultiLabelDataset(MultiLabelClassificationDataset):
    url = "https://drive.google.com/file/d/1DtKiauowCB0ykjFe8v0OVvT76rEfOk0v/view"

    labels = LABELS
    name = "UC Merced multilabel dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
