from marshmallow import fields, validate

from ..base.schemas import (
    BaseClassifierSchema,
    BaseFoundationModelSchema,
    BaseModelSchema,
    BaseSegmentationClassifierSchema,
)


class TransformerModelSchema(BaseClassifierSchema):
    """
    Schema for configuring a transformer model.
    """

    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={
            "description": "Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia"
        },
    )

    d_model = fields.Int(load_default=64, metadata={"description": "None"})

    n_head = fields.Int(required=False, load_default=2, metadata={"description": "None"})

    n_layers = fields.Int(required=False, load_default=5, metadata={"description": "None"})

    d_inner = fields.Int(required=False, load_default=128, metadata={"description": "None"})

    dropout = fields.Float(required=False, load_default=0.4, metadata={"description": "None"})

    activation = fields.String(
        required=False,
        load_default="relu",
        metadata={"description": "Which activation function to use"},
    )

    weight_decay = fields.Float(
        required=False,
        load_default=0.0000000552,
        metadata={"description": "Weight decay for Adam optimizer"},
    )


class InceptionTimeSchema(BaseClassifierSchema):
    """
    Schema for configuring a InceptionTime model.
    """

    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={
            "description": "Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia"
        },
    )

    num_layers = fields.Int(required=False, load_default=3, metadata={"description": "None"})

    hidden_dims = fields.Int(required=False, load_default=128, metadata={"description": "None"})

    use_bias = fields.Bool(required=False, load_default=False, metadata={"description": "None"})

    weight_decay = fields.Float(
        required=False,
        load_default=0.0000000552,
        metadata={"description": "Weight decay for Adam optimizer"},
    )


class LSTMSchema(BaseClassifierSchema):
    """
    Schema for configuring a LSTM model.
    """

    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={
            "description": "Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia"
        },
    )

    num_layers = fields.Int(required=False, load_default=4, metadata={"description": "None"})

    hidden_dims = fields.Int(required=False, load_default=128, metadata={"description": "None"})

    bidirectional = fields.Bool(required=False, load_default=True, metadata={"description": "None"})

    use_layernorm = fields.Bool(required=False, load_default=True, metadata={"description": "None"})

    dropout = fields.Float(
        required=False, load_default=0.5713020228087161, metadata={"description": "None"}
    )

    weight_decay = fields.Float(
        required=False,
        load_default=0.0000000552,
        metadata={"description": "Weight decay for Adam optimizer"},
    )


class MSResNetSchema(BaseClassifierSchema):
    """
    Schema for configuring a MSResNet model.
    """

    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={
            "description": "Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia"
        },
    )

    layers = fields.List(
        fields.Int, load_default=[1, 1, 1, 1], required=False, metadata={"description": "None"}
    )

    hidden_dims = fields.Int(required=False, load_default=32, metadata={"description": "None"})

    weight_decay = fields.Float(
        required=False,
        load_default=0.0000000552,
        metadata={"description": "Weight decay for Adam optimizer"},
    )


class TempCNNSchema(BaseClassifierSchema):
    """
    Schema for configuring a TempCNN model.
    """

    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={
            "description": "Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia"
        },
    )

    sequence_length = fields.Int(required=False, load_default=45, metadata={"description": "None"})

    hidden_dims = fields.Int(required=False, load_default=128, metadata={"description": "None"})

    kernel_size = fields.Int(required=False, load_default=7, metadata={"description": "None"})

    dropout = fields.Float(
        required=False, load_default=0.18203942949809093, metadata={"description": "None"}
    )

    weight_decay = fields.Float(
        required=False,
        load_default=0.0000000552,
        metadata={"description": "Weight decay for Adam optimizer"},
    )


class StarRNNSchema(BaseClassifierSchema):
    """
    Schema for configuring a StarRNN model.
    """

    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={
            "description": "Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia"
        },
    )

    num_layers = fields.Int(required=False, load_default=3, metadata={"description": "None"})

    hidden_dims = fields.Int(required=False, load_default=128, metadata={"description": "None"})

    bidirectional = fields.Bool(
        required=False, load_default=False, metadata={"description": "None"}
    )

    use_layernorm = fields.Bool(required=False, load_default=True, metadata={"description": "None"})

    use_batchnorm = fields.Bool(
        required=False, load_default=False, metadata={"description": "None"}
    )

    dropout = fields.Float(required=False, load_default=0.5, metadata={"description": "None"})

    weight_decay = fields.Float(
        required=False,
        load_default=0.0000000552,
        metadata={"description": "Weight decay for Adam optimizer"},
    )


class OmniScaleCNNSchema(BaseClassifierSchema):
    """
    Schema for configuring a OmniScaleCNN model.
    """

    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={
            "description": "Number of bands (13 for L1C, 10 for L2A), 11 for eopatch slovenia"
        },
    )

    sequence_length = fields.Int(required=False, load_default=45, metadata={"description": "None"})

    parameter_number_of_layer_list = fields.List(
        fields.Int,
        load_default=[8 * 128, 5 * 128 * 256 + 2 * 256 * 128],
        required=False,
        metadata={"description": "None"},
    )

    few_shot = fields.Bool(required=False, load_default=False, metadata={"description": "None"})

    hidden_dims = fields.Int(required=False, load_default=128, metadata={"description": "None"})

    kernel_size = fields.Int(required=False, load_default=7, metadata={"description": "None"})

    dropout = fields.Float(
        required=False, load_default=0.18203942949809093, metadata={"description": "None"}
    )

    weight_decay = fields.Float(
        required=False,
        load_default=0.0000000552,
        metadata={"description": "Weight decay for Adam optimizer"},
    )


class UnsupervisedDeepMulticlassClassifierSchema(BaseModelSchema):
    learning_rate = fields.Float(
        load_default=0.05, metadata={"description": "Learning rate used in training."}
    )
    weight_decay = fields.Float(
        load_default=-5, metadata={"description": "Weight decay for optimizer."}
    )
    number_of_clusters = fields.Integer(
        load_default=100, metadata={"description": "Number of clusters to use during traning."}
    )
    sobel = fields.Boolean(
        load_default=False, metadata={"description": "Whether to turn on on sobel filtering."}
    )


class UNetEfficientNetModelSchema(BaseSegmentationClassifierSchema):
    net = fields.String(
        required=True,
        metadata={"description": "Which neural network architecture to use"},
        validate=validate.OneOf(["b4", "b5", "b6", "b7", "l2"]),
    )
    stride = fields.Int(
        required=True,
        metadata={"description": "Controls how the filters convolve the input"},
        validate=validate.OneOf([16, 32]),
    )


class CNNRNNModelSchema(BaseModelSchema):
    learning_rate = fields.Float(
        required=False,
        load_default=0.0001,
        metadata={"description": "Learning rate used in training."},
    )
    embed_size = fields.Int(
        required=False,
        load_default=256,
        metadata={"description": "Dimension of decoder embedding vectors."},
    )
    hidden_size = fields.Int(
        required=False,
        load_default=512,
        metadata={"description": "Dimension of hidden LSTM states."},
    )
    num_layers = fields.Int(
        required=False, load_default=1, metadata={"description": "Number of LSTM layers."}
    )
    threshold = fields.Float(
        required=False, load_default=0.5, metadata={"description": "Label probability threshold."}
    )


class DeepLabV3ModelSchema(BaseSegmentationClassifierSchema):
    input_dim = fields.Int(
        required=True,
        validate=validate.OneOf([13, 10, 11]),
        metadata={"description": "Number of bands"},
    )


class AnySatSchema(BaseFoundationModelSchema):
    patch_size = fields.Integer(
        load_default=16, metadata={"description": "Patch size for the model input processing."}
    )

    output = fields.String(
        load_default="patch",
        validate=validate.OneOf(["patch", "tile", "dense", "all"]),
        metadata={
            "description": "Output format: 'patch' (tokens), 'dense' (segmentation map),'tile' (classification vector) or all."
        },
    )


class PrestoSchema(BaseFoundationModelSchema):
    pixel_batch_size = fields.Int(
        required=False,
        load_default=64,
        metadata={"description": "Batch size for pixel-wise processing."},
    )
    month = fields.Int(
        required=False,
        load_default=6,
        metadata={
            "description": "Default starting month (0-11) for time-series encoding if not provided in the input sample."
        },
    )


class TerraMindSchema(BaseFoundationModelSchema):
    modalities = fields.List(
        fields.String,
        load_default=None,
        metadata={
            "description": "List of modalities to be used as input.",
            "example": ["S2L2A", "S2L1C", "RGB", "S1GRD", "S1RTC", "DEM"],
        },
    )
    output_modalities = fields.List(
        fields.String,
        load_default=None,
        metadata={
            "description": "List of modalities to be used as output.",
            "example": ["S2L2A", "S2L1C", "RGB", "S1GRD", "S1RTC", "DEM"],
        },
    )
    tim_modalities = fields.List(
        fields.String,
        load_default=None,
        metadata={
            "description": "List of modalities to be used in Thinking in Modalities (TiM) setting.",
            "example": ["S2L2A", "S2L1C", "S1GRD", "S1RTC", "DEM"],
        },
    )
    merge_method = fields.String(
        load_default="mean",
        allow_none=True,
        validate=validate.OneOf(["mean", "max", "concat", "dict", None]),
        metadata={
            "description": "Method to merge output for further processing.",
            "example": "mean",
        },
    )
