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
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/tmp/",
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
    dir = fields.String(
        required=True,
        description="Directory with the image to perform prediction on",
        example="/tmp/test/",
    )
    model_path = fields.String(
        required=True,
        description="Path to the model",
        example="/tmp/model/checkpoint.pth.tar",
    )
    output_path = fields.String(
        missing="predictions.csv",
        description="File or folder path where the csv or plot predictions will be stored",
    )
    dataset_config = fields.Nested(
        missing=None,
        nested=ObjectConfig,
        description="Dataset type and configuration.",
    )
    labels = fields.Dict(
        missing=None, description="Labels needed to tag the predictions.",
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


class PredictLabelsTask(PredictTaskSchema):
    output_format = fields.String(
        missing="plot",
        description="Whether to output the predictions to csv or plots",
        validate=validate.OneOf(["plot", "csv"]),
    )


class PrepareTaskSchema(BaseTaskShema):
    dataset_config = fields.Nested(
        nested=ObjectConfig,
        required=True,
        description="Dataset type and configuration.",
    )


class ExtractFeaturesTaskSchema(BaseTaskShema):
    dir = fields.String(
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


class SpaceNet6PreprocessTaskSchema(BaseTaskShema):
    # TODO: Write descriptions
    root_directory = fields.String(required=True, description="The root directory of the downloaded data")
    ############################
    fold_directory = fields.String(required=True, description="The root directory of the downloaded data")
    segmentation_directory = fields.String(required=True, description="Some target directory")
    edge_width = fields.Int(required=True, description="description", default=9)
    contact_width = fields.Int(required=True, description="description", default=3)
    rotation_input_path = fields.String(required=True, description="some")
    rotation_output_path = fields.String(required=True, description="some")
    #############################
    num_threads = fields.Int(required=False, missing=1, description="Number of threads")


class SpaceNet6TrainAndEvaluateTaskSchema(BaseTaskShema):
    # TODO: Write descriptions
    loss_eps = fields.Float(required=True, description="some")
    focal_gamma = fields.Float(required=True, description="some")
    wd = fields.Float(required=True, description="some")
    lr = fields.Float(required=True, description="some")

    root_directory = fields.String(required=True, description="The root directory of the downloaded data")
    fold_path = fields.String(required=True, description="The root directory of the downloaded data")
    segmentation_directory = fields.String(required=True, description="Some target directory")

    train = fields.Bool(required=False, default=True, description="some")

    warm_up_dec_epochs = fields.Int(required=True, description="some")

    input_scale = fields.Float(required=True, description="some")
    direction_scale = fields.Float(required=True, description="some")
    coord_scale = fields.Float(required=True, description="some")

    b_count_weight = fields.Float(required=True, description="some")
    b_rev_size_weight = fields.Float(required=True, description="some")
    pos_weight = fields.Float(required=True, description="some")
    focal_weight = fields.Float(required=True, description="some")
    edge_weight = fields.Float(required=True, description="some")
    contact_weight = fields.Float(required=True, description="some")

    aux_scale = fields.Float(required=True, description="some")

    apex = fields.Bool(required=True, description="some")

    clip_grad_norm_value = fields.Float(required=True, description="some")

    # models_folder = fields.String(required=True, description="some")
    snapshot_last = fields.String(required=True, description="some")
    snapshot_best = fields.String(required=True, description="some")

    start_val_epoch = fields.Int(required=True, description="some")

    val = fields.Bool(required=True, description="some")
    test = fields.Bool(required=True, description="some")

    pred_folder = fields.String(required=True, description="some")
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