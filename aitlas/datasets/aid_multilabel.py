from .generic_multilabel import GenericMultiLabelsDataset


CLASSES_TO_IDX = {
    "airplane": 0,
    "bare-soil": 1,
    "buildings": 2,
    "cars": 3,
    "chaparral": 4,
    "court": 5,
    "dock": 6,
    "field": 7,
    "grass": 8,
    "mobile-home": 9,
    "pavement": 10,
    "sand": 11,
    "sea": 12,
    "ship": 13,
    "tanks": 14,
    "trees": 15,
    "water": 16,
}


class AIDMultiLabelDataset(GenericMultiLabelsDataset):
    url = "https://github.com/Hua-YS/AID-Multilabel-Dataset"

    classes_to_idx = CLASSES_TO_IDX
