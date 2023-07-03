from marshmallow import Schema, fields, validate

from ..base import ObjectConfig


class BaseTaskShema(Schema):
    """
    Schema for configuring a base task.

    :param log: Flag indicating whether to turn on logging. Default is True.
    :type log: bool, optional

    :param id: Run name/ID for the task. Default is None.
    :type id: str, optional
    """
    log = fields.Boolean(required=False, missing=True, description="Turn on logging")
    id = fields.String(
        required=False,
        description="Run name/ID",
        example="train_eurosat_123",
        missing=None,
    )


class SplitSetObjectSchema(Schema):
    """
    Schema for configuring a split dataset object.

    :param ratio: Ratio of the dataset to include in the split. This is required.
    :type ratio: int

    :param file: File containing the indices for the split. This is required.
    :type file: str
    """
    ratio = fields.Int(required=True, description="Ratio of dataset", example=60)
    file = fields.String(
        required=True, description="File indices", example="./data/indices.csv"
    )


class SplitObjectSchema(Schema):
    train = fields.Nested(SplitSetObjectSchema, required=True)
    val = fields.Nested(SplitSetObjectSchema, required=False, missing=None)
    test = fields.Nested(SplitSetObjectSchema, required=True)


class SplitTaskSchema(BaseTaskShema):
    """
    Schema for configuring a split task.

    :param data_dir: Path to the dataset on disk. This is required.
    :type data_dir: str

    :param csv_file: CSV file on disk containing dataset information. Default is None.
    :type csv_file: str, optional

    :param split: Configuration on how to split the dataset. Default is None.
    :type split: SplitObjectSchema, optional
    """
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
    """
    Schema for configuring a training task.

    :param dataset_config: Train dataset type and configuration. This is required.
    :type dataset_config: ObjectConfig

    :param epochs: Number of epochs used in training. This is required.
    :type epochs: int

    :param model_directory: Directory of the model output. This is required.
    :type model_directory: str

    :param save_epochs: Number of training steps between model checkpoints. Default is 100.
    :type save_epochs: int, optional

    :param iterations_log: After how many mini-batches do we want to show something in the log. Default is 200.
    :type iterations_log: int, optional

    :param resume_model: File path to the model to be resumed. Default is None.
    :type resume_model: str, optional
    """
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
    """
    Schema for configuring a task that involves training and evaluation.

    :param epochs: Number of epochs used in training. This is required.
    :type epochs: int

    :param model_directory: Directory of the model output. This is required.
    :type model_directory: str

    :param save_epochs: Number of training steps between model checkpoints. Default is 100.
    :type save_epochs: int, optional

    :param iterations_log: After how many mini-batches do we want to show something in the log. Default is 200.
    :type iterations_log: int, optional

    :param resume_model: File path to the model to be resumed. Default is None.
    :type resume_model: str, optional

    :param train_dataset_config: Train dataset type and configuration. This is required.
    :type train_dataset_config: ObjectConfig

    :param val_dataset_config: Validation dataset type and configuration. This is required.
    :type val_dataset_config: ObjectConfig
    """
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
    """
        Schema for configuring an optimization task.
    """
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
    """
    Schema for configuring an evaluation task.

    :param dataset_config: Dataset type and configuration. This is required.
    :type dataset_config: ObjectConfig

    :param model_path: Path to the model. This is required.
    :type model_path: str

    :param metrics: Metric classes you want to calculate. Default is an empty list.
    :type metrics: List[str], optional

    :param visualizations: Visualization classes you want to show. Default is an empty list.
    :type visualizations: List[str], optional
    """
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
    """
    Schema for configuring a prediction task.

    :param data_dir: Directory with the image to perform prediction on. This is required.
    :type data_dir: str

    :param model_path: Path to the model. This is required.
    :type model_path: str

    :param output_dir: Folder path where the plot images with predictions will be stored. Default is '/predictions'.
    :type output_dir: str, optional

    :param output_file: CSV file path where the predictions will be stored. Default is 'predictions.csv'.
    :type output_file: str, optional

    :param dataset_config: Dataset type and configuration. Default is None.
    :type dataset_config: ObjectConfig, optional

    :param batch_size: Batch size. Default is 64.
    :type batch_size: int, optional

    :param labels: Labels needed to tag the predictions. Default is None.
    :type labels: List[str], optional

    :param transforms: Classes to run transformations. Default is a list of common torchvision transformations.
    :type transforms: List[str], optional

    :param output_format: Whether to output the predictions to CSV or plots. Default is 'plot'.
                          Must be one of ['plot', 'csv', 'image'].
    :type output_format: str, optional
    """
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
    """
    Schema for configuring a task to extract features from images.
    """
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
