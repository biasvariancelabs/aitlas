from marshmallow import Schema, fields, validate


class BaseTaskShema(Schema):
    log = fields.Boolean(required=False, missing=True, description="Turn on logging")
    id = fields.String(
        required=False,
        description="Run name/ID",
        example="train_eurosat_123",
        missing=None,
    )


class TrainTaskSchema(BaseTaskShema):
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
    iterations_log = fields.Int(
        missing=200,
        description="After how many mini-batches do we want to show something in the log.",
    )
    resume_model = fields.String(
        missing=None,
        description="File path to the model to be resumed",
        example="/tmp/model/checkpoint.pth.tar",
    )


class EvaluateTaskSchema(BaseTaskShema):
    model_path = fields.String(
        required=True,
        description="Path to the model",
        example="/tmp/model/checkpoint.pth.tar",
    )
    metrics = fields.List(
        fields.String,
        missing=[],
        description="Metric classes you want to calculate",
        example=["aitlas.metrics.PrecisionScore", "aitlas.metrics.AccuracyScore"],
    )
    visualizations = fields.List(
        fields.String,
        missing=[],
        description="Visualizations classes you want to show",
        example=["aitlas.visualizations.ConfusionMatrix"],
    )


class SplitTaskSchema(BaseTaskShema):
    pass
