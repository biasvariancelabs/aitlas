from eolearn.core import FeatureType
from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema


_valid_types = [t.value for t in FeatureType]


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


class EOPatchDatasetSchema(BaseDatasetSchema):
    root = fields.String(
        missing="/", description="Dataset path on disk", example="./data/EOPatches/"
    )
    input_feature_type = fields.String(
        description="Feature type of the input feature.",
        required=True,
        validate=validate.OneOf(_valid_types),
    )
    input_feature_name = fields.String(
        description="Name of the input feature.", required=True
    )
    # input_feature_axis = fields.List(fields.Int, description="Height and width axis for the input features",
    #                                  required=True, example=[1, 2])
    input_feature_shape = fields.List(
        fields.Int,
        description="Shape of the input feature. Use -1 for unknown dimesnions.",
        required=True,
        example=[-1, 100, 100, 3],
    )

    labels_feature_type = fields.String(
        description="Feature type of the labels feature.",
        required=True,
        validate=validate.OneOf(_valid_types),
    )
    labels_feature_name = fields.String(
        description="Name of the labels feature.", required=True
    )
    # labels_feature_axis = fields.List(fields.Int, description="Height and width axis for the labels", required=True,
    #                                   example=[1, 2])
    labels_feature_shape = fields.List(
        fields.Int,
        description="Shape of the labels feature. Use -1 for unknown dimesnions.",
        required=True,
        example=[-1, 100, 100, 3],
    )
