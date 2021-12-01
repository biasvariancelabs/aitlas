from .multiclass_classification import MultiClassClassificationDataset


LABELS = [
    "airplane",
    "baseball_field",
    "basketball_court",
    "beach",
    "bridge",
    "cemetery",
    "chaparral",
    "christmas_tree_farm",
    "closed_road",
    "coastal_mansion",
    "crosswalk",
    "dense_residential",
    "ferry_terminal",
    "football_field",
    "forest",
    "freeway",
    "golf_course",
    "harbor",
    "intersection",
    "mobile_home_park",
    "nursing_home",
    "oil_gas_field",
    "oil_well",
    "overpass",
    "parking_lot",
    "parking_space",
    "railway",
    "river",
    "runway",
    "runway_marking",
    "shipping_yard",
    "solar_panel",
    "sparse_residential",
    "storage_tank",
    "swimming_pool",
    "tennis_court",
    "transformer_station",
    "wastewater_treatment_plant",
]


class PatternNetDataset(MultiClassClassificationDataset):

    url = "https://arxiv.org/abs/1706.03424"
    labels = LABELS

    name = "PatternNet dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
