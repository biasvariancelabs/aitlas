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
    dir_path = fields.String(
        required=True,
        description="Dataset path on disk",
        example="./data/tmp/ or ./data/tmp/images.csv",
    )
    csv_file_path = fields.String(
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


class SpaceNet5PrepareImageryTaskSchema(BaseTaskShema):
    in_dir = fields.String(
        required=True, description="Input directory containing the raw images"
    )
    out_dir = fields.String(
        required=True, description="Output directory for saving the processed images"
    )
    rescale_type = fields.String(
        required=True,
        description="The type of pixel value rescaling",
        validate=validate.OneOf(["clip", "perc", "dict"]),
    )
    percentiles = fields.String(
        required=False,
        missing="2,98",
        description="Percentile range to clip the pixel values",
    )
    band_order = fields.String(
        required=False,
        missing="5,3,2",
        description="Channel order for the R,G and B components",
    )
    num_threads = fields.Integer(
        required=False, missing=1, description="Number of threads"
    )


class SpaceNet5PrepareSpeedMasksTaskSchema(BaseTaskShema):
    geojson_dir = fields.String(
        required=True, description="Input directory containing the GeoJSON files"
    )
    image_dir = fields.String(
        required=True,
        description="Directory of input images (output of the PrepareImagery task)",
    )
    output_conversion_csv = fields.String(
        required=True,
        description="Output csv filepath for the speed<->burn value conversion",
    )
    output_mask_contin_dir = fields.String(
        required=True,
        description="Output directory for saving the continuous speed masks",
    )
    output_mask_multidim_dir = fields.String(
        required=True,
        description="Output directory for saving the multi-channel masks, "
        "leave it empty if you don't want to generate them",
    )
    buffer_distance_meters = fields.String(
        required=False,
        missing=2,
        description="Road buffer (in meters), "
        "assigned for each road center-line prediction",
    )


class SpaceNet5SkeletonizeTaskSchema(BaseTaskShema):
    min_spur_length_m = fields.Int(
        required=True, description="Minimum length (in meters) for edge filtering"
    )
    min_subgraph_length_pix = fields.Int(
        required=True, description="Minimum number of pixels for edge filtering"
    )
    # TODO: Write better description
    GSD = fields.Float(required=True, description="Some description")
    # TODO: Write better description
    use_medial_axis = fields.Int(required=True, description="Some description")
    path_results_root = fields.String(
        required=True, description="Root directory for the output files"
    )
    test_results_dir = fields.String(
        required=True, description="Directory for the results from this task"
    )
    wkt_submission = fields.String(
        required=True, description="CSV file with a list of road segments"
    )
    skeleton_dir = fields.String(
        required=True, description="Directory for the resulting skeleton masks"
    )
    skeleton_pkl_dir = fields.String(
        required=True, description="Directory for the pickled resulting skeleton masks"
    )
    masks_dir = fields.String(
        required=True,
        description="Directory with the input masks which are to be skeletonized",
    )
    skeleton_thresh = fields.Float(
        required=True, description="Threshold for skeletonization"
    )
    skeleton_band = fields.Int(
        required=True, description="Which channel to use to extract a mask skeleton"
    )
    num_classes = fields.Int(
        required=True, description="The number of classes (i.e. output channels)"
    )


class SpaceNet5WktToGraphTaskSchema(BaseTaskShema):
    min_subgraph_length_pix = fields.Int(
        required=True, description="Minimum length (in meters) for edge filtering"
    )
    min_spur_length_m = fields.Int(
        required=True, description="Minimum number of pixels for edge filtering"
    )
    path_results_root = fields.String(
        required=True, description="Root directory for the output files"
    )
    test_results_dir = fields.String(
        required=True, description="Directory for the results from this task"
    )
    test_data_refined_dir = fields.String(
        required=True,
        description="Directory of input images (output of the PrepareImagery task)",
    )
    wkt_submission = fields.String(
        required=True,
        description="CSV file with a list of road segments (output of the Skeletonize task)",
    )
    graph_dir = fields.String(
        required=True, description="Output directory for the graph pickles"
    )
    num_channels = fields.Int(
        required=False, missing=3, description="Number of input channels"
    )
    # TODO: Write better description
    rdp_epsilon = fields.Int(required=False, missing=1, description="rdp epsilon")


class SpaceNet5InferSpeedTaskSchema(BaseTaskShema):
    path_results_root = fields.String(
        required=True, description="Root directory for the output files"
    )
    test_results_dir = fields.String(
        required=True, description="Directory for the results from this task"
    )
    test_data_refined_dir = fields.String(
        required=True,
        description="Directory of input images (output of the PrepareImagery task)",
    )
    graph_dir = fields.String(
        required=True, description="Input directory for the graph pickles"
    )
    num_channels = fields.Int(required=True, description="Number of input channels")
    speed_conversion_file = fields.String(
        required=True, description="Output of the PrepareMasks task"
    )
    masks_dir = fields.String(
        required=True, description="Directory with the target masks"
    )
    skeleton_band = fields.Int(
        required=True, description="Which channel to use to extract a mask skeleton"
    )
    num_classes = fields.Int(
        required=True, description="The number of classes (i.e. output channels)"
    )


class SpaceNet5PlotGraphOverImageTaskSchema(BaseTaskShema):
    path_results_root = fields.String(
        required=True, description="Root directory for the output files"
    )
    test_results_dir = fields.String(
        required=True, description="Directory for the results from this task"
    )
    test_data_refined_dir = fields.String(
        required=True,
        description="Directory of input images (output of the PrepareImagery task)",
    )
    graph_dir = fields.String(
        required=True, description="Input directory for the graph pickles"
    )
