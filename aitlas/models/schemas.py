from marshmallow import fields

from ..base import BaseClassifierSchema


class ResnetSchema(BaseClassifierSchema):
    pretrained = fields.Bool(
        missing=False, description="Whether to use a pretrained network or not."
    )


class FastRCNNSchema(BaseClassifierSchema):
    pretrained = fields.Bool(
        missing=False, description="Whether to use a pretrained network or not."
    )


class DeepLabV3Schema(BaseClassifierSchema):
    pretrained = fields.Bool(
        missing=False, description="Whether to use a pretrained network or not."
    )
