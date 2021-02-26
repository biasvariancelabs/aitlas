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


class SpaceNet6DatasetSchema(BaseDatasetSchema):
    train = fields.Bool(required=True, description="some")
    color = fields.Bool(required=True, description="some")

    crop_size = fields.Int(required=True, description="some")

    reorder_bands = fields.Int(required=True, description="some")

    rot_prob = fields.Float(required=True, description="some")
    scale_prob = fields.Float(required=True, description="some")
    color_aug_prob = fields.Float(required=True, description="some")
    gamma_aug_prob = fields.Float(required=True, description="some")
    gauss_aug_prob = fields.Float(required=True, description="some")
    elastic_aug_prob = fields.Float(required=True, description="some")
    flipud_prob = fields.Float(required=True, description="some")
    fliplr_prob = fields.Float(required=True, description="some")
    rot90_prob = fields.Float(required=True, description="some")
    channel_swap_prob = fields.Float(required=True, description="some")

    train_min_building_size = fields.Float(required=True, description="some")

    rot_out_path = fields.String(required=True, description="some")
    root_directory = fields.String(required=True, description="some")
