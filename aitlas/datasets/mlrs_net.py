import glob

import pandas as pd

from .multilabel_classification import MultiLabelClassificationDataset


LABELS = [
    "airplane",
    "airport",
    "bare soil",
    "baseball diamond",
    "basketball court",
    "beach",
    "bridge",
    "buildings",
    "cars",
    "cloud",
    "containers",
    "crosswalk",
    "dense residential area",
    "desert",
    "dock",
    "factory",
    "field",
    "football field",
    "forest",
    "freeway",
    "golf course",
    "grass",
    "greenhouse",
    "gully",
    "habor",
    "intersection",
    "island",
    "lake",
    "mobile home",
    "mountain",
    "overpass",
    "park",
    "parking lot",
    "parkway",
    "pavement",
    "railway",
    "railway station",
    "river",
    "road",
    "roundabout",
    "runway",
    "sand",
    "sea",
    "ships",
    "snow",
    "snowberg",
    "sparse residential area",
    "stadium",
    "swimming pool",
    "tanks",
    "tennis court",
    "terrace",
    "track",
    "trail",
    "transmission tower",
    "trees",
    "water",
    "chaparral",
    "wetland",
    "wind turbine",
]


class MLRSNetMultiLabelDataset(MultiLabelClassificationDataset):
    url = "https://data.mendeley.com/datasets/7j9bv9vwsx/2"

    labels = LABELS
    name = "MLRSNet dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)


# Function to convert the dataset in PASCAL VOC data format
# First unrar all the images in the images folder using this command for linux: for file in *.rar; do unrar e "$file"; done
def prepare(root_folder):
    all_csv_filenames = [
        i for i in glob.glob("{}{}/*.{}".format(root_folder, "labels", "csv"))
    ]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_csv_filenames])
    combined_csv["image"] = combined_csv["image"].str.replace(".jpg", "", regex=False)
    combined_csv.to_csv(
        "{}/multilabels.txt".format(root_folder),
        index=False,
        sep="\t",
        encoding="utf-8",
    )
