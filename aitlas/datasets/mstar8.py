from .multiclass_classification import MultiClassClassificationDataset

'''
The Moving and Stationary Target Acquisition and Recognition (MSTAR) was a joint Defense Advanced Research 
Projects Agency (DARPA) and Air Force Research Laboratory (AFRL) effort to develop and evaluate anadvanced 
ATR system. The program began in June of 1995 and ended in 1999. The MSTAR-8class dataset provides 12166 SAR 
image chips covering 8 target types from the former Soviet Union. The images were collected by the Sandia 
National Laboratory (SNL) using the STARLOS sensor.
'''

LABELS = ["2S1","BRDM_2","BTR_60","D7","SLICY","T62","ZIL131","ZSU_23_4"]


class MSTAR8Dataset(MultiClassClassificationDataset):

    url = "https://www.kaggle.com/datasets/atreyamajumdar/mstar-dataset-8-classes/data"
    labels = LABELS
    name = "MSATR-8 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)