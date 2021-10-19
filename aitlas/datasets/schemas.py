from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema


class MultiClassClassificationDatasetSchema(BaseDatasetSchema):
    download = fields.Bool(
        missing=False, description="Whether to download the dataset", example=True
    )
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )


class MultiLabelClassificationDatasetSchema(BaseDatasetSchema):
    root = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )


class SegmentationDatasetSchema(BaseDatasetSchema):
    root = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )


class BigEarthNetSchema(BaseDatasetSchema):
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    lmdb_path = fields.String(required=True, description="Path to the lmdb storage")
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    import_to_lmdb = fields.Bool(
        missing=False, description="Should the data be moved to LMDB"
    )
    bands10_mean = fields.List(
        fields.Float,
        missing=[429.9430203, 614.21682446, 590.23569706],
        required=False,
        description="List of mean values for the 3 channels",
    )
    bands10_std = fields.List(
        fields.Float,
        missing=[572.41639287, 582.87945694, 675.88746967],
        required=False,
        description="List of std values for the 3 channels",
    )

class CropsDatasetSchema(BaseDatasetSchema):
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    root = fields.String(
        required=True, description="Dataset path on disk", example="./slovenia-crops"
    )
    year = fields.Integer(
        missing=2017, description="year", validate=validate.OneOf([2017, 2018])
    )
    verbose = fields.Bool(
        missing=False, description="verbose" #change to true
    )    
    level = fields.String(
        missing='L1C', description="L1C or L2A", example="L1C", validate=validate.OneOf(['L1C', 'L2A'])
    )
    regions = fields.List(
        fields.String,
        required=True, description="Brittany region (frh01..frh04) or train/val/test", example="['frh01','frh01']")

class BreizhCropsSchema(CropsDatasetSchema):

 
    filter_length = fields.Integer(
        missing=0, description="filter_length"
    )
    load_timeseries = fields.Bool(
        missing=True, description="load_timeseries"
    )    
    recompile_h5_from_csv = fields.Bool(
        missing=False, description="recompile_h5_from_csv"
    )    
    preload_ram = fields.Bool(
        missing=False, description="preload_ram"
    )