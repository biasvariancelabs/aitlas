from marshmallow import fields

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
    train = fields.Bool(missing=True, description="Is it train dataset", example=True)
