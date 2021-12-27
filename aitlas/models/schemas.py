from aitlas.base.schemas import BaseClassifierSchema
from marshmallow import Schema, fields, validate

from ..base import BaseModelSchema
from ..base.schemas import BaseSegmentationClassifierSchema


class TransformerModelSchema(BaseClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia",
        validate=validate.OneOf([13, 10, 11]),
    )

    d_model = fields.Int(missing=64, description="None")

    n_head = fields.Int(required=False, missing=2, description="None")

    n_layers = fields.Int(required=False, missing=5, description="None")

    d_inner = fields.Int(required=False, missing=128, description="None")

    dropout = fields.Float(required=False, missing=0.4, description="None")

    activation = fields.String(
        required=False, missing="relu", description="Which activation function to use"
    )

    weight_decay = fields.Float(
        required=False,
        missing=0.0000000552,
        description="Weight decay for Adam optimizer",
    )


class InceptionTimeSchema(BaseClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia",
        validate=validate.OneOf([13, 10, 11]),
    )

    num_layers = fields.Int(required=False, missing=3, description="None")

    hidden_dims = fields.Int(required=False, missing=128, description="None")

    use_bias = fields.Bool(required=False, missing=False, description="None")

    weight_decay = fields.Float(
        required=False,
        missing=0.0000000552,
        description="Weight decay for Adam optimizer",
    )


class LSTMSchema(BaseClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia",
        validate=validate.OneOf([13, 10, 11]),
    )

    num_layers = fields.Int(required=False, missing=4, description="None")

    hidden_dims = fields.Int(required=False, missing=128, description="None")

    bidirectional = fields.Bool(required=False, missing=True, description="None")

    use_layernorm = fields.Bool(required=False, missing=True, description="None")

    dropout = fields.Float(
        required=False, missing=0.5713020228087161, description="None"
    )

    weight_decay = fields.Float(
        required=False,
        missing=0.0000000552,
        description="Weight decay for Adam optimizer",
    )


class MSResNetSchema(BaseClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia",
        validate=validate.OneOf([13, 10, 11]),
    )

    layers = fields.List(
        fields.Int, missing=[1, 1, 1, 1], required=False, description="None"
    )

    hidden_dims = fields.Int(required=False, missing=32, description="None")

    weight_decay = fields.Float(
        required=False,
        missing=0.0000000552,
        description="Weight decay for Adam optimizer",
    )


class TempCNNSchema(BaseClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia",
        validate=validate.OneOf([13, 10, 11]),
    )

    sequence_length = fields.Int(required=False, missing=45, description="None")

    hidden_dims = fields.Int(required=False, missing=128, description="None")

    kernel_size = fields.Int(required=False, missing=7, description="None")

    dropout = fields.Float(
        required=False, missing=0.18203942949809093, description="None"
    )

    weight_decay = fields.Float(
        required=False,
        missing=0.0000000552,
        description="Weight decay for Adam optimizer",
    )


class StarRNNSchema(BaseClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia",
        validate=validate.OneOf([13, 10, 11]),
    )

    num_layers = fields.Int(required=False, missing=3, description="None")

    hidden_dims = fields.Int(required=False, missing=128, description="None")

    bidirectional = fields.Bool(required=False, missing=False, description="None")

    use_layernorm = fields.Bool(required=False, missing=True, description="None")

    use_batchnorm = fields.Bool(required=False, missing=False, description="None")

    dropout = fields.Float(required=False, missing=0.5, description="None")

    weight_decay = fields.Float(
        required=False,
        missing=0.0000000552,
        description="Weight decay for Adam optimizer",
    )


class OmniScaleCNNSchema(BaseClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia",
        validate=validate.OneOf([13, 10, 11]),
    )

    sequence_length = fields.Int(required=False, missing=45, description="None")

    parameter_number_of_layer_list = fields.List(
        fields.Int,
        missing=[8 * 128, 5 * 128 * 256 + 2 * 256 * 128],
        required=False,
        description="None",
    )

    few_shot = fields.Bool(required=False, missing=False, description="None")

    hidden_dims = fields.Int(required=False, missing=128, description="None")

    kernel_size = fields.Int(required=False, missing=7, description="None")

    dropout = fields.Float(
        required=False, missing=0.18203942949809093, description="None"
    )

    weight_decay = fields.Float(
        required=False,
        missing=0.0000000552,
        description="Weight decay for Adam optimizer",
    )


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


class CNNRNNModelSchema(BaseModelSchema):
    learning_rate = fields.Float(
        required=False, missing=0.0001, description="Learning rate used in training."
    )
    embed_size = fields.Int(
        required=False,
        missing=256,
        description="Dimension of decoder embedding vectors.",
    )
    hidden_size = fields.Int(
        required=False, missing=512, description="Dimension of hidden LSTM states."
    )
    num_layers = fields.Int(
        required=False, missing=1, description="Number of LSTM layers."
    )
    threshold = fields.Float(
        required=False, missing=0.5, description="Label probability threshold."
    )
