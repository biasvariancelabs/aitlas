from marshmallow import fields

from aitlas.base.schemas import BaseClassifierSchema

import sys
sys.path.append("/e/AiTLAS/aitlas/ESRI-challenge/models")

class BaseDetectionClassifierSchema(BaseClassifierSchema):
    metrics = fields.List(
        fields.String,
        missing=["models.metrics.mAP"],
        description="Classes of metrics you want to calculate",
        example=["aitlas.metrics.mAP"],
    )