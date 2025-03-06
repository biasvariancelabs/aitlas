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
    url = "https://www.dropbox.com/scl/fi/dlrn78p72m1wxdyc1y2l9/UCMercedLanduse.zip?rlkey=hsu2tnwvszcb2i7ewoyuvt34p&e=2"

    labels = LABELS
    name = "UC Merced multilabel dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
