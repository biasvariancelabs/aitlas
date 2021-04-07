from marshmallow import fields, validate

from aitlas.base.schemas import BaseClassifierSchema


class TransformerModelSchema(BaseClassifierSchema):

    input_dim = fields.Int(required=True,
                        description="Number of bands (13 for L1C, 10 for L2A)",
                        validate=validate.OneOf([13, 10]))

    d_model = fields.Int(missing = 64,
                        description="None")

    n_head = fields.Int(required=False, missing = 2,
                        description="None")

    n_layers = fields.Int(required=False, missing = 5,
                        description="None")

    d_inner = fields.Int(required=False, missing = 128,
                        description="None")

    dropout = fields.Float(required=False, missing = 0.017998950510888446,
                        description="None")

    activation = fields.String(required=False, missing="relu",
                        description="Which activation function to use")
