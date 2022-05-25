from marshmallow import Schema, fields, validate

from ..base import ObjectConfig


class BaseTaskShema(Schema):
    log = fields.Boolean(required=False, missing=True, description="Turn on logging")
    id = fields.String(
        required=False,
        description="Run name/ID",
        example="train_eurosat_123",
        missing=None,
    )


class SplitSetObjectSchema(Schema):
    ratio = fields.Int(required=True, description="Ratio of dataset", example=60)
    file = fields.String(
        required=True, description="File indices", example="./data/indices.csv"
    )


class SplitObjectSchema(Schema):
    train = fields.Nested(SplitSetObjectSchema, required=True)
    val = fields.Nested(SplitSetObjectSchema, required=False, missing=None)
    test = fields.Nested(SplitSetObjectSchema, required=True)


class SplitTaskSchema(BaseTaskShema):
    data_dir = fields.String(
        required=True,
        description="Dataset path on disk",
        example="./data/tmp/ or ./data/tmp/images.csv",
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )
    split = fields.Nested(
        SplitObjectSchema,
        description="Configuration on how to split the dataset.",
        missing=None,
    )


class TrainTaskSchema(BaseTaskShema):
    dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Train dataset type and configuration.",
    )
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


class TrainAndEvaluateTaskSchema(BaseTaskShema):
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
    train_dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Train dataset type and configuration.",
    )
    val_dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Validation dataset type and configuration.",
    )


class ParameterSchema(Schema):
    name = fields.String(required=True, description="Parameter to optimize")
    min = fields.Float(missing=0, description="Lower end of range.",)
    max = fields.Float(missing=0.5, description="Higher end of range.",)
    steps = fields.Int(
        missing=10, description="In how mane steps to iterate the range",
    )


class OptimizeTaskSchema(BaseTaskShema):
    epochs = fields.Int(
        required=True, description="Number of epochs used in training", example=50
    )
    model_directory = fields.String(
        required=True,
        description="Directory of the model output",
        example="/tmp/model/",
    )
    train_dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Train dataset type and configuration.",
    )
    val_dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Validation dataset type and configuration.",
    )
    parameters = fields.Nested(
        ParameterSchema,
        required=True,
        many=True,
        description="Parameters to optimize.",
    )
    method = fields.String(
        required=True,
        description="How to search through the ranges: grid or random",
        example="grid",
        validate=validate.OneOf(["grid", "random"]),
    )


class EvaluateTaskSchema(BaseTaskShema):
    dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Dataset type and configuration.",
    )
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


class PredictTaskSchema(BaseTaskShema):
    data_dir = fields.String(
        required=True,
        description="Directory with the image to perform prediction on",
        example="/tmp/test/",
    )
    model_path = fields.String(
        required=True,
        description="Path to the model",
        example="/tmp/model/checkpoint.pth.tar",
    )
    output_dir = fields.String(
        missing="/predictions",
        description="Folder path where the plot images with predictions will be stored",
    )
    output_file = fields.String(
        missing="predictions.csv",
        description="CSV file path where the predictions will be stored",
    )
    dataset_config = fields.Nested(
        missing=None,
        nested=ObjectConfig,
        description="Dataset type and configuration.",
    )
    batch_size = fields.Int(missing=64, description="Batch size", example=64)
    labels = fields.List(
        fields.String,
        missing=None,
        description="Labels needed to tag the predictions.",
    )
    transforms = fields.List(
        fields.String,
        missing=[
            "torchvision.transforms.ToPILImage",
            "torchvision.transforms.Resize",
            "torchvision.transforms.CenterCrop",
            "torchvision.transforms.ToTensor",
        ],
        description="Classes to run transformations.",
    )
    output_format = fields.String(
        missing="plot",
        description="Whether to output the predictions to csv or plots",
        validate=validate.OneOf(["plot", "csv", "image"]),
    )


class PrepareTaskSchema(BaseTaskShema):
    dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Dataset type and configuration.",
    )


class ExtractFeaturesTaskSchema(BaseTaskShema):
    data_dir = fields.String(
        required=True,
        description="Directory with images to extract features from",
        example="/tmp/test/",
    )
    output_dir = fields.String(
        missing="predictions.csv",
        description="Folder path where the features will be saved",
    )
    model_path = fields.String(
        missing=None,
        description="Path to the model that will generate the features",
        example="/tmp/model/extractor.pth.tar",
    )
    transforms = fields.List(
        fields.String,
        missing=[
            "torchvision.transforms.ToPILImage",
            "torchvision.transforms.Resize",
            "torchvision.transforms.CenterCrop",
            "torchvision.transforms.ToTensor",
        ],
        description="Classes to run transformations.",
    )


class VisualizeSplitSetObjectSchema(Schema):
    dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Dataset type and configuration.",
    )


class VisualizeSplitObjectSchema(Schema):
    train = fields.Nested(ObjectConfig, required=False, missing=None)
    val = fields.Nested(ObjectConfig, required=False, missing=None)
    test = fields.Nested(ObjectConfig, required=False, missing=None)


class VisualizeTaskSchema(BaseTaskShema):
    output_xls = fields.String(
        missing=None, description="Excel file path where the splits will be saved",
    )
    output_file = fields.String(
        missing="plot.jpg", description="Image file path where the plots will be shown",
    )
    split = fields.Nested(
        VisualizeSplitObjectSchema,
        description="Configuration with the splits to the dataset.",
        missing=None,
    )
