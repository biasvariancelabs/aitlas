from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema, CsvDatasetSchema, SplitableDatasetSchema


class RootFolderSchema(SplitableDatasetSchema):
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/dataset/",
    )


class EurosatDatasetSchema(SplitableDatasetSchema):
    download = fields.Bool(
        missing=True, description="Whether to download the dataset", example=True
    )
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/EuroSAT/"
    )
    mode = fields.String(
        missing="rgb",
        default="Work with rgb or all bands mode",
        example="rgb",
        validate=validate.OneOf(["rgb", "all"]),
    )


class CrackForestSchema(SplitableDatasetSchema):
    root = fields.String(
        required=True,
        description="Dataset path on disk",
        example="./data/CrackForest-dataset-master/",
    )


class UcMercedDatasetSchema(SplitableDatasetSchema):
    download = fields.Bool(
        missing=True, description="Whether to download the dataset", example=True
    )
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/UcMerced/"
    )


class UcMercedMultiLabelsDatasetSchema(SplitableDatasetSchema):
    download = fields.Bool(
        missing=True, description="Whether to download the dataset", example=True
    )
    root = fields.String(
        required=True,
        description="Is it train dataset",
        example="./data/UcMercedMultiLabels/",
    )


class Resisc45DatasetSchema(SplitableDatasetSchema):
    download = fields.Bool(
        missing=True, description="Whether to download the dataset", example=True
    )
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/Resisc45/"
    )


class PatternNetDatasetSchema(SplitableDatasetSchema):
    download = fields.Bool(
        missing=True, description="Whether to download the dataset", example=True
    )
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/PatternNet/"
    )


class BigEarthNetSchema(CsvDatasetSchema):
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
