from marshmallow import fields

from ..base import BaseClassifierSchema


class ResnetSchema(BaseClassifierSchema):
    pretrained = fields.Bool(
        missing=False, description="Whether to use a pretrained network or not."
    )
