from .multilabel_classification import MultiLabelClassificationDataset

LABELS = ["impervious", "water", "clutter", "vegetation", "building", "tree", "boat", "car"]


class DFC15MultiLabelDataset(MultiLabelClassificationDataset):
    url = "https://github.com/Hua-YS/DFC15-Multilabel-Dataset"

    labels = LABELS
