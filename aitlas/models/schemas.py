from marshmallow import fields

from aitlas.base.schemas import BaseSegmentationClassifierSchema


class UNetEfficientNetModelSchema(BaseSegmentationClassifierSchema):
    # TODO: Write better descriptions
    net = fields.String(required=True, description="some", example="b5")
    stride = fields.Int(required=True, description="some")
    extra_num = fields.Int(required=True, description="some")
    glob = fields.Bool(required=True, description="some")
    bn = fields.Bool(required=True, description="some")
    aspp = fields.Bool(required=True, description="some")
    ocr = fields.Bool(required=True, description="some")
    aux = fields.Bool(required=True, description="some")
    box1x1 = fields.Bool(required=True, description="some")
