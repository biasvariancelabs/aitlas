from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema


class MatDatasetSchema(BaseDatasetSchema):
    mat_file_path = fields.String(
        missing=None, description="mat file on disk", example="./data/dataset.mat",
    )
    mode = fields.String(
        missing="train",
        description="Which split to use, train or test.",
        example="train",
    )
    download = fields.Bool(
        missing=False, description="Whether to download the dataset", example=True
    )


class MultiClassClassificationDatasetSchema(BaseDatasetSchema):
    download = fields.Bool(
        missing=False, description="Whether to download the dataset", example=True
    )
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )


class MultiLabelClassificationDatasetSchema(BaseDatasetSchema):
    root = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )


class SegmentationDatasetSchema(BaseDatasetSchema):
    root = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )


class BigEarthNetSchema(BaseDatasetSchema):
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    lmdb_path = fields.String(required=True, description="Path to the lmdb storage")
    root = fields.String(
        required=True, description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    import_to_lmdb = fields.Bool(
        missing=False, description="Should the data be moved to LMDB"
    )
    bands10_mean = fields.List(
        fields.Float,
        missing=[429.9430203, 614.21682446, 590.23569706],
        required=False,
        description="List of mean values for the 3 channels",
    )
    bands10_std = fields.List(
        fields.Float,
        missing=[572.41639287, 582.87945694, 675.88746967],
        required=False,
        description="List of std values for the 3 channels",
    )
class DotaDatasetSchema(BaseDatasetSchema):
    root = fields.String(
        required = True, description="Dataset path on disk", example = './data/DOTA' 
    )
    subset = fields.String(
        required = True, description = 'Which subset of the data is loaded', example = 'train_split'
    ) 
    subsample_percentage = fields.Float(
        missing = 0.1, description = '...', example = 0.1
    )
    filter_null = fields.Bool(
        missing = True, description = 'Whether to remove examples with no detectable objects', example = 'True')

class EsriChallengeDatasetSchema(BaseDatasetSchema):
    root = fields.String(
        required = True, description="Dataset path on disk", example = './data/DOTA' 
    )
    subset = fields.String(
        required = True, description = 'Which subset of the data is loaded', example = 'train_split'
    ) 
    subsample_percentage = fields.Float(
        missing = 0.1, description = '...', example = 0.1
    )


class SpaceNet6DatasetSchema(BaseDatasetSchema):
    orients = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Roterdam/SummaryData/SAR_orientations.csv",
        description="Absolute path pointing to the SAR orientations text file "
        "(output of the pre-processing task",
    )
    root_directory = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Rotterdam/",
        description="Root directory for the raw SpaceNet6 data set",
    )
    start_val_epoch = fields.Int(
        required=False,
        description="From which epoch should the validation period start",
    )
    # Train & val
    folds_path = fields.String(
        required=False,
        example="path/to/results/folds",
        description="Path to the fold csv files",
    )
    segmentation_directory = fields.String(
        required=False,
        example="path/to/results/segmentation",
        description="Source directory with the target segmentation masks",
    )
    gt_csv = fields.String(
        required=False,
        description="Source file containing the ground truth segmentation data on the buildings",
    )
    pred_csv = fields.String(
        required=False,
        description="Destination file for saving the predictions from the current fold",
    )
    pred_folder = fields.String(
        required=False,
        description="Destination directory for saving the predictions from all folds",
    )
    edge_weight = fields.Int(
        required=False, description="Weight for the building edges pixels"
    )
    contact_weight = fields.Int(
        required=False, description="Weight for the building contact pixels"
    )
    # Test
    test_directory = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Rotterdam/",
        description="Root directory for the raw SpaceNet6 data set",
    )
    merged_pred_dir = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Rotterdam/",
        description="Destination directory for merging the predictions from all folds",
    )
    solution_file = fields.String(
        required=False,
        example="path/to/data/results/solution.csv",
        description="SpaceNet6-compliant csv destination file used for grading the challenge",
    )
    # Prepare
    num_folds = fields.Int(
        required=False, missing=10, description="Number of fold splits for the data set"
    )
    orients_output = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Roterdam/SummaryData/SAR_orientations.txt",
        description="Absolute path pointing to the output SAR orientations csv file",
    )
    num_threads = fields.Int(
        required=False,
        missing=1,
        description="Number of threads for parallel execution",
    )
    edge_width = fields.Int(
        required=False,
        default=3,
        description="Width of the edge of buildings (in pixels)",
    )
    contact_width = fields.Int(
        required=False,
        default=9,
        description="Width of the contact between (in pixels)",
    )
    folds_dir = fields.String(
        required=False,
        example="path/to/results/folds",
        description="Source directory with the fold csv files",
    )


class BreizhCropsSchema(BaseDatasetSchema):
    regions = fields.List(
        fields.String,
        required=True,
        description="Brittany region (frh01..frh04)",
        example="['frh01','frh01']",
    )

    root = fields.String(
        required=True,
        description="Dataset path on disk",
        example="./breizhcrops_dataset",
    )
    year = fields.Integer(
        missing=2017, description="year", validate=validate.OneOf([2017, 2018])
    )
    filter_length = fields.Integer(missing=0, description="filter_length")
    level = fields.String(
        required=True,
        description="L1C or L2A",
        example="L1C",
        validate=validate.OneOf(["L1C", "L2A"]),
    )
    verbose = fields.Bool(missing=False, description="verbose")  # change to true
    load_timeseries = fields.Bool(missing=True, description="load_timeseries")
    recompile_h5_from_csv = fields.Bool(
        missing=False, description="recompile_h5_from_csv"
    )
    preload_ram = fields.Bool(missing=False, description="preload_ram")


class CropsDatasetSchema(BaseDatasetSchema):
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    root = fields.String(
        required=True, description="Dataset path on disk", example="./slovenia-crops"
    )
    year = fields.Integer(
        missing=2017, description="year", validate=validate.OneOf([2017, 2018])
    )
    verbose = fields.Bool(missing=False, description="verbose")  # change to true
    level = fields.String(
        missing="L1C",
        description="L1C or L2A",
        example="L1C",
        validate=validate.OneOf(["L1C", "L2A"]),
    )
    regions = fields.List(
        fields.String,
        required=True,
        description="Brittany region (frh01..frh04) or train/val/test",
        example="['frh01','frh01']",
    )


class BreizhCropsSchema(CropsDatasetSchema):

    filter_length = fields.Integer(missing=0, description="filter_length")
    load_timeseries = fields.Bool(missing=True, description="load_timeseries")
    recompile_h5_from_csv = fields.Bool(
        missing=False, description="recompile_h5_from_csv"
    )
    preload_ram = fields.Bool(missing=False, description="preload_ram")
