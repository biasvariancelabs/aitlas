from .multiclass_classification import MultiClassClassificationDataset

'''
The dataset was created to detect the fire smoke in satellite imagery. Smoke detection 
based on satellite imagery is imperative for wildfire detection and monitoring.
'''

LABELS = ["Cloud","Dust","Haze","Land","Seaside","Smoke"]


class USTCSmokeRSDataset(MultiClassClassificationDataset):

    url = "http://complex.ustc.edu.cn/2019/0802/c18202a389656/page.htm"
    labels = LABELS
    name = "USTC_SmokeRS dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)