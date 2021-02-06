from .generic_multilabel import GenericMultiLabelsDataset


CLASSES_TO_IDX = {
    "impervious": 0,
    "water": 1,
    "clutter": 2,
    "vegetation": 3,
    "building": 4,
    "tree": 5,
    "boat": 6,
    "car": 7

}


class DFC15MultiLabelDataset(GenericMultiLabelsDataset):
    url = "https://github.com/Hua-YS/DFC15-Multilabel-Dataset"

    classes_to_idx = CLASSES_TO_IDX
