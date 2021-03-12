from .multiclass_classification import MultiClassClassificationDataset


LABELS = ["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral", "denseresidential",
          "forest", "freeway", "golfcourse", "harbor", "intersection", "mediumresidential", "mobilehomepark",
          "overpass", "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"]


class UcMercedDataset(MultiClassClassificationDataset):

    labels = LABELS
