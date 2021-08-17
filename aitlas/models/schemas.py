from marshmallow import Schema, fields

from ..base import BaseModelSchema
from ..base.schemas import BaseSegmentationClassifierSchema


class UnsupervisedDeepMulticlassClassifierSchema(BaseModelSchema):
    learning_rate = fields.Float(
        missing=0.05, description="Learning rate used in training."
    )
    weight_decay = fields.Float(missing=-5, description="Weight decay for optimizer.")
    number_of_clusters = fields.Integer(
        missing=100, description="Number of clusters to use during traning."
    )
    sobel = fields.Boolean(
        missing=False, description="Whether to turn on on sobel filtering."
    )


class UNetEfficientNetModelSchema(BaseSegmentationClassifierSchema):
    net = fields.String(
        required=True,
        description="Which neural network architecture to use",
        validate=validate.OneOf(["b4", "b5", "b6", "b7", "l2"]),
    )
    stride = fields.Int(
        required=True,
        description="Controls how the filters convolve the input",
        validate=validate.OneOf([16, 32]),
    )
