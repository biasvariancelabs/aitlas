from marshmallow import Schema, fields, validate
from ..base.schemas import BaseModelSchema
from ..base.schemas import BaseClassifierSchema
from ..base.schemas import BaseSegmentationClassifierSchema
from ..base.schemas import BaseFoundationModelSchema


class TransformerModelSchema(BaseClassifierSchema):
    """
    Schema for configuring a transformer model.
    """
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
    """
    Schema for configuring a InceptionTime model.
    """
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
    """
    Schema for configuring a LSTM model.
    """
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
    """
    Schema for configuring a MSResNet model.
    """
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
    """
    Schema for configuring a TempCNN model.
    """
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
    """
    Schema for configuring a StarRNN model.
    """
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
    """
    Schema for configuring a OmniScaleCNN model.
    """
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

class DeepLabV3ModelSchema(BaseSegmentationClassifierSchema):
    input_dim = fields.Int(
        required=True,
        description="Number of bands",
        validate=validate.OneOf([13, 10, 11]),
    )

class PrestoSchema(BaseFoundationModelSchema):
    pixel_batch_size = fields.Int(
        required=False,
        load_default=64,
        description="Batch size for pixel-wise processing."
    )
    month = fields.Int(
        required=False,
        load_default=6,
        description="Default starting month (0-11) for time-series encoding if not provided in the input sample.",
    )

class TerraMindSchema(BaseFoundationModelSchema):
    modalities = fields.List(
        fields.String,
        missing=None,
        description="List of modalities to be used as input.",
        example=['S2L2A', 'S2L1C', 'RGB', 'S1GRD', 'S1RTC', 'DEM']
    )
    output_modalities = fields.List(
        fields.String,
        missing=None,
        description="List of modalities to be used as output.",
        example=['S2L2A', 'S2L1C', 'RGB', 'S1GRD', 'S1RTC', 'DEM']
    )
    tim_modalities = fields.List(
        fields.String,
        missing=None,
        description="List of modalities to be used in Thinking in Modalities (TiM) setting.",
        example=['S2L2A', 'S2L1C', 'S1GRD', 'S1RTC', 'DEM']
    )
