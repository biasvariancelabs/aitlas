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

class MSResNetSchema(BaseClassifierSchema):

    input_dim = fields.Int(required=True,
                        description="Number of bands (13 for L1C, 10 for L2A)",
                        validate=validate.OneOf([13, 10]))

    layers = fields.List(fields.Int, missing = [1, 1, 1, 1], required=False,
                        description="None")

    hidden_dims = fields.Int(required=False, missing = 32,
                        description="None")

class TempCNNSchema(BaseClassifierSchema):

    input_dim = fields.Int(required=True,
                        description="Number of bands (13 for L1C, 10 for L2A)",
                        validate=validate.OneOf([13, 10]))

    sequence_length = fields.Int(required=False, missing = 45,
                        description="None")

    hidden_dims = fields.Int(required=False, missing = 128,
                        description="None")

    kernel_size = fields.Int(required=False, missing = 7,
                        description="None")

    dropout = fields.Float(required=False, missing = 0.18203942949809093,
                        description="None")

class StarRNNSchema(BaseClassifierSchema):

    input_dim = fields.Int(required=True,
                        description="Number of bands (13 for L1C, 10 for L2A)",
                        validate=validate.OneOf([13, 10]))

    num_layers = fields.Int(required=False, missing = 3,
                        description="None")

    hidden_dims = fields.Int(required=False, missing = 128,
                        description="None")

    bidirectional = fields.Bool(required=False, missing = False,
                        description="None")

    use_layernorm = fields.Bool(required=False, missing = True,
                        description="None")

    use_batchnorm = fields.Bool(required=False, missing = False,
                        description="None")

    dropout = fields.Float(required=False, missing = 0.5,
                        description="None")

class OmniScaleCNNSchema(BaseClassifierSchema):

    input_dim = fields.Int(required=True,
                        description="Number of bands (13 for L1C, 10 for L2A)",
                        validate=validate.OneOf([13, 10]))

    sequence_length = fields.Int(required=False, missing = 45,
                        description="None")

    parameter_number_of_layer_list = fields.List(fields.Int, missing = [8 * 128, 5 * 128 * 256 + 2 * 256 * 128], required=False,
                        description="None")

    few_shot = fields.Bool(required=False, missing = False,
                        description="None")

    hidden_dims = fields.Int(required=False, missing = 128,
                        description="None")

    kernel_size = fields.Int(required=False, missing = 7,
                        description="None")

    dropout = fields.Float(required=False, missing = 0.18203942949809093,
                        description="None")
