from marshmallow import fields, validate

from aitlas.base.schemas import BaseSegmentationClassifierSchema


class UNetEfficientNetModelSchema(BaseSegmentationClassifierSchema):
    net = fields.String(required=True,
                        description="Which neural network architecture to use",
                        validate=validate.OneOf(["b4", "b5", "b6", "b7", "l2"]))
    stride = fields.Int(required=True,
                        description="Controls how the filters convolve the input",
                        validate=validate.OneOf([16, 32]))
