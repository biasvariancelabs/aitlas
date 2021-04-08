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


class InceptionTimeSchema(BaseClassifierSchema):

    input_dim = fields.Int(required=True,
                        description="Number of bands (13 for L1C, 10 for L2A)",
                        validate=validate.OneOf([13, 10]))

    num_layers = fields.Int(required=False, missing = 4,
                        description="None")

    hidden_dims = fields.Int(required=False, missing = 64,
                        description="None")

    use_bias = fields.Bool(required=False, missing = False,
                        description="None")


class LSTMSchema(BaseClassifierSchema):

    input_dim = fields.Int(required=True,
                        description="Number of bands (13 for L1C, 10 for L2A)",
                        validate=validate.OneOf([13, 10]))

    num_layers = fields.Int(required=False, missing = 4,
                        description="None")

    hidden_dims = fields.Int(required=False, missing = 128,
                        description="None")

    bidirectional = fields.Bool(required=False, missing = True,
                        description="None")

    use_layernorm = fields.Bool(required=False, missing = True,
                        description="None")

    dropout = fields.Float(required=False, missing = 0.5713020228087161,
                        description="None")
