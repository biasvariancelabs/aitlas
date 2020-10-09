from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema, SplitableDatasetSchema


class CifarDatasetSchema(BaseDatasetSchema):
    download = fields.Bool(
        missing=True, description="Whether to download the dataset", example=True
    )
    train = fields.Bool(missing=True, description="Is it train dataset", example=True)


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


class PennFundanSchema(SplitableDatasetSchema):
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/PennFundan/"
    )
    download = fields.Bool(
        missing=True, description="Whether to download the dataset", example=True
    )


class CrackForestSchema(SplitableDatasetSchema):
    root = fields.String(
        required=True,
        description="Dataset path on disk",
        example="./data/CrackForest-dataset-master/",
    )
