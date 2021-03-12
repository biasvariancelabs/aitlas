from .multilabel_classification import MultiLabelClassificationDataset

LABELS = ["airplane", "bare-soil", "buildings", "cars", "chaparral", "court", "dock", "field", "grass",
          "mobile-home", "pavement", "sand", "sea", "ship", "tanks", "trees", "water"]


class AIDMultiLabelDataset(MultiLabelClassificationDataset):
    url = "https://github.com/Hua-YS/AID-Multilabel-Dataset"

    labels = LABELS
