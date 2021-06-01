from marshmallow import Schema, fields

from ..base import BaseModelSchema


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


class CNNRNNModelSchema(BaseModelSchema):
    learning_rate = fields.Float(
        required=False, missing=0.0001, description="Learning rate used in training."
    )
    embed_size = fields.Int(
        required=False, missing=256, description="Dimension of decoder embedding vectors."
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
