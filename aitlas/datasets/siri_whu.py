from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "agriculture",
    "commercial",
    "harbor",
    "idle_land",
    "industrial",
    "meadow",
    "overpass",
    "park",
    "pond",
    "residential",
    "river",
    "water",
]


class SiriWhuDataset(MultiClassClassificationDataset):

    url = "http://www.lmars.whu.edu.cn/prof_web/zhongyanfei/e-code.html"
    labels = LABELS
    name = "SIRI-WHU dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
