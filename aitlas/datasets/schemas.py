from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema


class MatDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring a classification dataset given as mat file.
    """

    mat_file = fields.String(
        load_default=None,
        metadata={"description": "mat file on disk", "example": "./data/dataset.mat"},
    )
    mode = fields.String(
        load_default="train",
        metadata={"description": "Which split to use, train or test.", "example": "train"},
    )
    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    download = fields.Bool(
        load_default=False,
        metadata={"description": "Whether to download the dataset", "example": True},
    )


class NPZDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring a classification dataset given as npz file.
    """

    npz_file = fields.String(
        load_default=None,
        metadata={"description": "npz file on disk", "example": "./data/dataset.npz"},
    )
    mode = fields.String(
        load_default="train",
        metadata={"description": "Which split to use, train or test.", "example": "train"},
    )
    labels = fields.List(
        fields.String,
        load_default=None,
        required=False,
        metadata={"description": "List of labels"},
    )


class ClassificationDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring a classification dataset.
    """

    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/BigEarthNet/"},
    )
    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )


class SegmentationDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring a segmentation dataset.
    """

    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/BigEarthNet/"},
    )
    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )


class ObjectDetectionPascalDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in PASCAL VOC format.
    """

    imageset_file = fields.String(
        load_default="/",
        metadata={
            "description": "File with the image ids in the set",
            "example": "./data/DIOR/train.txt",
        },
    )
    image_dir = fields.String(
        load_default="/",
        metadata={"description": "Folder to the images on disk", "example": "./data/DIOR/"},
    )
    annotations_dir = fields.String(
        load_default="/",
        metadata={
            "description": "Folder with the XML annotations in VOC format",
            "example": "./data/DIOR/Annons/",
        },
    )


class ObjectDetectionCocoDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in COCO format.
    """

    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/DIOR/"},
    )
    json_file = fields.String(
        load_default=None,
        metadata={"description": "JSON Coco file format on disk", "example": "./data/train.json"},
    )
    hardcode_background = fields.Bool(
        load_default=True,
        metadata={"description": "Do we need to hardcode the background as a class?"},
    )


class ObjectDetectionRfiCocoDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in COCO format.
    """

    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/DIOR/"},
    )
    annotation_dir = fields.String(
        load_default="/",
        metadata={"description": "JSON Coco files path on disk", "example": "./data/annotations/"},
    )


class BigEarthNetSchema(BaseDatasetSchema):
    """
    Schema for configuring the BigEarthNet dataset.
    """

    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    lmdb_path = fields.String(
        load_default=None, metadata={"description": "Path to the lmdb storage"}
    )
    data_dir = fields.String(
        load_default=None,
        metadata={"description": "Dataset path on disk", "example": "./data/BigEarthNet/"},
    )
    selection = fields.String(
        load_default="rgb",
        metadata={"description": "Read RGB channels or 13 channels", "example": "all/rgb"},
    )
    version = fields.String(
        load_default="19 labels",
        metadata={"description": "43 or 19 labels", "example": "43 labels/19 labels"},
    )
    import_to_lmdb = fields.Bool(
        load_default=False, metadata={"description": "Should the data be moved to LMDB"}
    )
    bands10_mean = fields.List(
        fields.Float,
        load_default=(429.9430203, 614.21682446, 590.23569706),
        required=False,
        metadata={"description": "List of mean values for the 3 channels"},
    )
    bands10_std = fields.List(
        fields.Float,
        load_default=(572.41639287, 582.87945694, 675.88746967),
        required=False,
        metadata={"description": "List of std values for the 3 channels"},
    )


class SpaceNet6DatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the SpaceNet6 dataset.
    """

    orients = fields.String(
        required=False,
        metadata={
            "description": "Absolute path pointing to the SAR orientations text file "
            "(output of the pre-processing task",
            "example": "path/to/data/train/AOI_11_Roterdam/SummaryData/SAR_orientations.csv",
        },
    )
    root_directory = fields.String(
        required=False,
        metadata={
            "description": "Root directory for the raw SpaceNet6 data set",
            "example": "path/to/data/train/AOI_11_Rotterdam/",
        },
    )
    start_val_epoch = fields.Int(
        required=False,
        metadata={"description": "From which epoch should the validation period start"},
    )
    # Train & val
    folds_path = fields.String(
        required=False,
        metadata={"description": "Path to the fold csv files", "example": "path/to/results/folds"},
    )
    segmentation_directory = fields.String(
        required=False,
        metadata={
            "description": "Source directory with the target segmentation masks",
            "example": "path/to/results/segmentation",
        },
    )
    gt_csv = fields.String(
        required=False,
        metadata={
            "description": "Source file containing the ground truth segmentation data on the buildings"
        },
    )
    pred_csv = fields.String(
        required=False,
        metadata={
            "description": "Destination file for saving the predictions from the current fold"
        },
    )
    pred_folder = fields.String(
        required=False,
        metadata={"description": "Destination directory for saving the predictions from all folds"},
    )
    edge_weight = fields.Int(
        required=False, metadata={"description": "Weight for the building edges pixels"}
    )
    contact_weight = fields.Int(
        required=False, metadata={"description": "Weight for the building contact pixels"}
    )
    # Test
    test_directory = fields.String(
        required=False,
        metadata={
            "description": "Root directory for the raw SpaceNet6 data set",
            "example": "path/to/data/train/AOI_11_Rotterdam/",
        },
    )
    merged_pred_dir = fields.String(
        required=False,
        metadata={
            "description": "Destination directory for merging the predictions from all folds",
            "example": "path/to/data/train/AOI_11_Rotterdam/",
        },
    )
    solution_file = fields.String(
        required=False,
        metadata={
            "description": "SpaceNet6-compliant csv destination file used for grading the challenge",
            "example": "path/to/data/results/solution.csv",
        },
    )
    # Prepare
    num_folds = fields.Int(
        required=False,
        load_default=10,
        metadata={"description": "Number of fold splits for the data set"},
    )
    orients_output = fields.String(
        required=False,
        metadata={
            "description": "Absolute path pointing to the output SAR orientations csv file",
            "example": "path/to/data/train/AOI_11_Roterdam/SummaryData/SAR_orientations.txt",
        },
    )
    num_threads = fields.Int(
        required=False,
        load_default=1,
        metadata={"description": "Number of threads for parallel execution", "example": "1"},
    )
    edge_width = fields.Int(
        required=False,
        load_default=3,
        metadata={"description": "Width of the edge of buildings (in pixels)", "example": "3"},
    )
    contact_width = fields.Int(
        required=False,
        load_default=9,
        metadata={"description": "Width of the contact between (in pixels)", "example": "9"},
    )
    folds_dir = fields.String(
        required=False,
        metadata={
            "description": "Source directory with the fold csv files",
            "example": "path/to/results/folds",
        },
    )


class BreizhCropsSchema(BaseDatasetSchema):
    """
    Schema for configuring the BreizhCrops dataset for crop type prediction.
    """

    regions = fields.List(
        fields.String,
        required=True,
        metadata={"description": "Brittany region (frh01..frh04)", "example": "['frh01','frh01']"},
    )

    root = fields.String(
        required=True,
        metadata={"description": "Dataset path on disk", "example": "./breizhcrops_dataset"},
    )
    year = fields.Integer(
        load_default=2017,
        validate=validate.OneOf([2017, 2018]),
        metadata={"description": "year", "example": "2017"},
    )
    filter_length = fields.Integer(
        load_default=0,
        metadata={"description": "filter_length"},
    )
    level = fields.String(
        required=True,
        validate=validate.OneOf(["L1C", "L2A"]),
        metadata={"description": "L1C or L2A", "example": "L1C"},
    )
    verbose = fields.Bool(load_default=False, metadata={"description": "verbose"})
    load_timeseries = fields.Bool(load_default=True, metadata={"description": "load_timeseries"})
    recompile_h5_from_csv = fields.Bool(
        load_default=False, metadata={"description": "recompile_h5_from_csv"}
    )
    preload_ram = fields.Bool(load_default=False, metadata={"description": "preload_ram"})


class CropsDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring dataset for crop type prediction.
    """

    csv_file_path = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    root = fields.String(
        required=True,
        metadata={"description": "Dataset path on disk", "example": "./slovenia-crops"},
    )
    verbose = fields.Bool(load_default=False, metadata={"description": "verbose"})
    level = fields.String(
        load_default="L1C",
        validate=validate.OneOf(["L1C", "L2A"]),
        metadata={"description": "L1C or L2A", "example": "L1C"},
    )
    regions = fields.List(
        fields.String,
        required=True,
        metadata={
            "description": "Brittany region (frh01..frh04) or train/val/test",
            "example": "['frh01','frh01']",
        },
    )


class So2SatDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the So2Sat dataset.
    """

    h5_file = fields.String(
        required=True, metadata={"description": "H5 file on disk", "example": "./data/train.h5"}
    )


class CloudDatasets_AI4QCSchema(BaseDatasetSchema):
    """
    Schema for configuring Cloud AI4QC datasets.
    """

    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/CESBIO/"},
    )
    selection = fields.String(
        load_default="rgb",
        metadata={"description": "Read RGB channels or 13 channels", "example": "all/rgb"},
    )


class ClusteringDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring a clustering dataset.
    """

    data_dir = fields.String(
        load_default="/",
        metadata={"description": "Dataset path on disk", "example": "./data/New_anomalies/"},
    )
    csv_file = fields.String(
        load_default=None,
        metadata={"description": "CSV file on disk", "example": "./data/train.csv"},
    )
