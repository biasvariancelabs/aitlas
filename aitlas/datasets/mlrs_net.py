import glob
import pandas as pd
from .multilabel_classification import MultiLabelClassificationDataset

LABELS = ["airplane", "airport", "bareland", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral",
          "cloud", "commercial_area", "dense_residential_area", "desert", "eroded_farmland", "farmland", "forest",
          "freeway", "golf_course", "ground_track_field", "harbor&port", "industrial_area", "intersection", "island",
          "lake", "meadow", "mobile_home_park", "mountain", "overpass", "park", "parking_lot", "parkway", "railway",
          "railway_station", "river", "roundabout", "shipping_yard", "snowberg", "sparse_residential_area", "stadium",
          "storage_tank", "swimmimg_pool", "tennis_court", "terrace", "transmission_tower", "vegetable_greenhouse",
          "wetland", "wind_turbine"]


class MLRSNet(MultiLabelClassificationDataset):
    url = "https://data.mendeley.com/datasets/7j9bv9vwsx/2"

    labels = LABELS
    name = "MLRSNet dataset"

    def prepare(self):
        all_csv_filenames = [i for i in glob.glob('/{}*.{}'.format(self.config.root, "csv"))]
        combined_csv = pd.concat([pd.read_csv(f) for f in all_csv_filenames])
        combined_csv.to_csv("{}/multilabels.txt".format(self.config.root), index=False, encoding='utf-8-sig')
        return True









