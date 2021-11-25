from .multilabel_classification import MultiLabelClassificationDataset


LABELS = [
    "impervious",
    "water",
    "clutter",
    "vegetation",
    "building",
    "tree",
    "boat",
    "car",
]


class DFC15MultiLabelDataset(MultiLabelClassificationDataset):
    url = "https://github.com/Hua-YS/DFC15-Multilabel-Dataset"

    labels = LABELS
    name = "DFC15 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
