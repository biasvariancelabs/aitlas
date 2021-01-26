from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema


class GenericMulticlassDatasetSchema(BaseDatasetSchema):
    download = fields.Bool(
        missing=False, description="Whether to download the dataset", example=True
    )
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )
    labels = fields.Dict(
        missing=None, description="Labels needed to tag the predictions.",
    )


class GenericMultiLabelsDatasetSchema(BaseDatasetSchema):
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
    transforms = fields.List(
        fields.String,
        missing=["aitlas.transforms.BaseSegmentation"],
        description="Classes to run transformations.",
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


class SpaceNet5DatasetSchema(BaseDatasetSchema):
    filenames = fields.String(
        missing="",
        description="File path to the {train, test, val}.csv output file by the SpaceNet5SplitTask"
    )
    image_root = fields.String(
        missing="", description="File path to the root directory with the output images by the PrepareImageryTask"
    )
    cont_mask_root = fields.String(
        missing="",
        description="File path to the root directory with the output continuous masks by the PrepareSpeedMasksTask"
    )
    mc_mask_root = fields.String(
        missing="",
        description="File path to the root directory with the output multi-channel masks by the PrepareSpeedMasksTask"
    )
