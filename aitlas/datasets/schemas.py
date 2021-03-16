from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema


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


class SpaceNet6DatasetSchema(BaseDatasetSchema):
    orients = fields.String(required=True,
                            example="path/to/data/train/AOI_11_Roterdam/SummaryData/SAR_orientations.csv",
                            description="Absolute path pointing to the SAR orientations text file "
                                        "(output of the pre-processing task")
    root_directory = fields.String(required=False,
                                   example="path/to/data/train/AOI_11_Rotterdam/",
                                   description="Root directory for the raw SpaceNet6 data set")
    start_val_epoch = fields.Int(required=False,
                                 description="From which epoch should the validation period start")
    # Train & val
    folds_path = fields.String(required=True,
                               example="path/to/results/folds",
                               description="Source directory with the fold csv files")
    segmentation_directory = fields.String(required=True,
                                           example="path/to/results/segmentation",
                                           description="Source directory with the target segmentation masks")
    gt_csv = fields.String(required=True,
                           description="Source file containing the ground truth segmentation data on the buildings")
    pred_csv = fields.String(required=True,
                             description="Destination file for saving the predictions from the current fold")
    pred_folder = fields.String(required=True,
                                description="Destination directory for saving the predictions from all folds")
    edge_weight = fields.Int(required=True,
                             description="Weight for the building edges pixels")
    contact_weight = fields.Int(required=True,
                                description="Weight for the building contact pixels")
    # Test
    test_directory = fields.String(required=False,
                                   example="path/to/data/train/AOI_11_Rotterdam/",
                                   description="Root directory for the raw SpaceNet6 data set")
    merged_pred_dir = fields.String(required=True,
                                    example="path/to/data/train/AOI_11_Rotterdam/",
                                    description="Destination directory for merging the predictions from all folds")
    solution_file = fields.String(required=True,
                                  example="path/to/data/results/solution.csv",
                                  description="SpaceNet6-compliant csv destination file used for grading the challenge")
