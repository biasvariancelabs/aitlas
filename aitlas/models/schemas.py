from marshmallow import Schema, fields

from ..base import BaseModelSchema


class UnsupervisedDeepMulticlassClassifierSchema(BaseModelSchema):
    learning_rate = fields.Float(
        missing=0.05, description="Learning rate used in training."
    )
    weight_decay = fields.Float(missing=-5, description="Weight decay for optimizer.")
    number_of_clusters = fields.Integer(
        missing=100, description="Number of clusters to use during traning."
    )
