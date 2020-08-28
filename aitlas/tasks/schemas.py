from marshmallow import Schema, fields, validate

from ..utils import CLASSIFICATION_METRICS


class TrainTaskSchema(Schema):
    epochs = fields.Int(
        required=True, description="Number of epochs used in training", example=50
    )
    model_directory = fields.String(
        required=True,
        description="Directory of the model output",
        example="/tmp/model/",
    )
    save_epochs = fields.Int(
        missing=100, description="Number of training steps between model checkpoints."
    )
    resume_model = fields.String(
        missing=None,
        description="File path to the model to be resumed",
        example="/tmp/model/checkpoint.pth.tar",
    )


class EvaluateTaskSchema(Schema):
    model_path = fields.String(
        required=True,
        description="Path to the model",
        example="/tmp/model/checkpoint.pth.tar",
    )
    metrics = fields.List(
        fields.String,
        missing=["f1_score"],
        description="Metrics you want to calculate",
        example=["accuracy", "precision", "recall", "f1_score"],
        validate=validate.ContainsOnly(list(CLASSIFICATION_METRICS.keys())),
    )


class SplitTaskSchema(Schema):
    pass
