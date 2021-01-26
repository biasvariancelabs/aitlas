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
    in_dir = fields.String(required=True, description="Input directory containing the raw images")
    out_dir = fields.String(required=True,
                            description="Output directory for saving the processed images")
    rescale_type = fields.String(required=True, description="The type of pixel value rescaling",
                                 validate=validate.OneOf(["clip", "perc", "dict"]))
    percentiles = fields.String(required=False, missing="2,98", description="Percentile range to clip the pixel values")
    band_order = fields.String(required=False, missing="5,3,2",
                               description="Channel order for the R,G and B components")
    num_threads = fields.Integer(required=False, missing=1, description="Number of threads")


class SpaceNet5PrepareSpeedMasksTaskSchema(BaseTaskShema):
    geojson_dir = fields.String(required=True, description="Input directory containing the GeoJSON files")
    image_dir = fields.String(required=True,
                              description="Directory of input images (output of the PrepareImagery task)")
    output_conversion_csv_binned = fields.String(required=True, description="Some description")
    output_mask_dir_contin = fields.String(required=True,
                                           description="Output directory for saving the continuous speed masks")
    output_mask_multidim_dir = fields.String(required=True,
                                             description="Output directory for saving the multi-channel masks, "
                                                         "leave it empty if you don't want to generate them")
    # TODO: Write better description
    buffer_distance_meters = fields.String(required=False, missing=2, description="Road buffer")


class SpaceNet5SkeletonizeTaskSchema(BaseTaskShema):
    # TODO: Write better descriptions
    min_spur_length_m = fields.Int(required=True, description="Some description")
    min_subgraph_length_pix = fields.Int(required=True, description="Some description")
    GSD = fields.Float(required=True, description="Some description")
    use_medial_axis = fields.Int(required=True, description="Some description")
    path_results_root = fields.String(required=True, description="Root directory for the output files")
    test_results_dir = fields.String(required=True, description="Directory for the results from this task")
    wkt_submission = fields.String(required=True,
                                   description="CSV file with a list of road segments")
    skeleton_dir = fields.String(required=True, description="Directory for the resulting skeleton masks")
    skeleton_pkl_dir = fields.String(required=True, description="Directory for the pickled resulting skeleton masks")
    masks_dir = fields.String(required=True, description="Directory with the masks which are to be skeletonized")
    skeleton_thresh = fields.Float(required=True, description="Threshold for skeletonization")
    skeleton_band = fields.Int(required=True, description="Bands")
    num_classes = fields.Int(required=True, description="The number of classes")


class SpaceNet5WktToGraphTaskSchema(BaseTaskShema):
    # TODO: Write better descriptions
    min_subgraph_length_pix = fields.Int(required=True, description="Some description")
    min_spur_length_m = fields.Int(required=True, description="Some description")
    path_results_root = fields.String(required=True, description="Root directory for the output files")
    test_results_dir = fields.String(required=True, description="Directory for the results from this task")
    test_data_refined_dir = fields.String(required=True,
                                          description="Directory of input images (output of the PrepareImagery task)")
    wkt_submission = fields.String(required=True,
                                   description="CSV file with a list of road segments (output of the Skeletonize task)")
    graph_dir = fields.String(required=True, description="Output directory for the graph pickles")
    num_channels = fields.Int(required=True, description="Num channels")
    rdp_epsilon = fields.Int(required=True, description="rdp epsilon")


class SpaceNet5InferSpeedTaskSchema(BaseTaskShema):
    # TODO: Write better descriptions
    path_results_root = fields.String(required=True, description="Root directory for the output files")
    test_results_dir = fields.String(required=True, description="Directory for the results from this task")
    test_data_refined_dir = fields.String(required=True,
                                          description="Directory of input images (output of the PrepareImagery task)")
    graph_dir = fields.String(required=True, description="Input directory for the graph pickles")
    num_channels = fields.Int(required=True, description="Num channels")
    speed_conversion_file = fields.String(required=True, description="Output of the PrepareMasks task")
    masks_dir = fields.String(required=True, description="Directory with the target masks")
    skeleton_band = fields.Int(required=True, description="Bands")
    num_classes = fields.Int(required=True, description="The number of classes")


class SpaceNet5PlotGraphOverImageTaskSchema(BaseTaskShema):
    # TODO: Set descriptions
    some_field = fields.String(required=True, description="Some description")
