from .multiclass_classification import MultiClassClassificationDataset


LABELS = ["Airplane", "Airport", "Artificial dense forest land", "Artificial sparse forest land", "Bare land",
          "Basketball court", "Blue structured factory building", "Building", "Construction site", "Cross river bridge",
          "Crossroads", "Dense tall building", "Dock", "Fish pond", "Footbridge", "Graff", "Grassland",
          "Low scattered building", "Lrregular farmland", "Medium density scattered building",
          "Medium density structured building", "Natural dense forest land", "Natural sparse forest land", "Oiltank",
          "Overpass", "Parking lot", "Plasticgreenhouse", "Playground", "Railway", "Red structured factory building",
          "Refinery", "Regular farmland", "Scattered blue roof factory building", "Scattered red roof factory building",
          "Sewage plant-type-one", "Sewage plant-type-two", "Ship", "Solar power station", "Sparse residential area",
          "Square", "Steelsmelter", "Storage land", "Tennis court", "Thermal power plant", "Vegetable plot", "Water"]


class RSD46WHUDataset(MultiClassClassificationDataset):

    url = "https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU"
    labels = LABELS

    name = "RSD46-WHU dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        MultiClassClassificationDataset.__init__(self, config)




