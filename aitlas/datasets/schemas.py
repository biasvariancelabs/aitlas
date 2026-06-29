from marshmallow import fields, validate
from ..base.schemas import BaseDatasetSchema

class ClassificationDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring a classification dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )


class SegmentationDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring a segmentation dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )


class ObjectDetectionPascalDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in PASCAL VOC format.
    """
    imageset_file = fields.String(
        missing="/",
        description="File with the image ids in the set",
        example="./data/DIOR/train.txt",
    )
    image_dir = fields.String(
        missing="/", description="Folder to the images on disk", example="./data/DIOR/"
    )
    annotations_dir = fields.String(
        missing="/",
        description="Folder with the XML annotations in VOC format",
        example="./data/DIOR/Annons/",
    )
    csv_file = fields.String(
        missing=None,
        description="CSV file containing the list of images for the split",
        example="./data/train.csv",
    )


class ObjectDetectionCocoDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in COCO format.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    json_file = fields.String(
        missing=None,
        description="JSON Coco file format on disk",
        example="./data/train.json",
    )
    hardcode_background = fields.Bool(
        missing=True, description="Do we need to hardcode the background as a class?"
    )


class ObjectDetectionRfiCocoDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in COCO format.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    annotation_dir = fields.String(
        missing='/',
        description="JSON Coco files path on disk",
        example="./data/annotations/"
    )


class FAIREOSchema(BaseDatasetSchema):
    """
    Schema for configuring FAIR-EO datasets.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )

class FAIREOObjectDetectionYoloSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in YOLO format.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    annotation_dir = fields.String(
        missing='/',
        description="txt Yolo files path on disk",
        example="./data/annotations/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    task = fields.String(
        missing=None, description="training, validation or test task", example="train"
    )

class FAIREOObjectDetectionPascalSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in PASCAL VOC format.
    """
    image_dir = fields.String(
        missing="/", description="Folder to the images on disk", example="./data/DIOR/"
    )
    annotations_dir = fields.String(
        missing="/",
        description="Folder with the XML annotations in VOC format",
        example="./data/DIOR/Annons/",
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )

class FAIREOObjectDetectionCocoSchema(BaseDatasetSchema):
    """
    Schema for configuring an object detection dataset given in COCO format.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    annotation_dir = fields.String(
        missing='/', description="json COCO files path on disk", example="./data/annotations/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    selection = fields.String(
        missing=None, description="Include all images or only images with labels", example="all"
    )

class MatDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring datasets given in .mat format.
    """
    mat_file = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    mode = fields.String(
	missing=None, description="train/val/test", example="train"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )

class LLCCDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the LLCC image classification dataset which has triplets of images.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )
    selection = fields.String(
        missing="/", description="Use images as triplets or as individual images", example="individual"
    )
    task = fields.String(
        missing=None, description="training, validation or test task", example="training"
    )

class OSMDatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the OSM dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    classification = fields.String(
        missing="/", description="Choose the classification schema (land_use, water_and_roads, buildings)", example="land_use"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )

class SpaceNet8DatasetSchema(BaseDatasetSchema):
    """
    Schema for configuring the SpaceNet8 dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./Germany_Training_Public/"
    )
    mode = fields.String(
        missing="post-event",
        description="Use only post-event images or pre-/post-event pairs",
        example="pairs",
    )

class SACropTypeSchema(BaseDatasetSchema):
    """
    Schema for configuring SA Crop Type dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or 13 channels", example="all/rgb"
    )
    date = fields.Int(
        missing=None, description="Read only one image out of a time series", example=1
    )

class MTS12Schema(BaseDatasetSchema):
    """
    Schema for configuring the MTS12 dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    date = fields.String(
        missing=None, description="Visualise images for a particular date", example="1"
    )  
    imagery = fields.String(
        missing="s2", description="Select which Sentinel imagery to use: 's1' for Sentinel-1, 's2' for Sentinel-2", example="s2"
    )
    split = fields.String(
        missing="train", description="Select which split to use, train, val or test", example="train"
    )

class Sen4AgriNetSchema(BaseDatasetSchema):
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    date = fields.String(
        missing=None, description="Visualise images for a particular date", example="1"
    )
    class_selection = fields.String(
        missing = "12", description = "Select the 12 core classes or all 168 classes", example="12/all"
    )

class BrazilDAMSchema(BaseDatasetSchema):
    """
    Schema for configuring the BrazilDAM dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    imagery = fields.String(
        missing="sentinel", description="Select which imagery to use: 'sentinel', 'landsat' or 'sentinel and landsat'", example="sentinel"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels, all channels or specific ones", example="all/rgb/bands"
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B01","B02","B03"]
    )
    bands_l8 = fields.List(
        fields.String(), missing=None, description="Select which L8 bands to use with a list of band names", example=["B01","B02","B03"]
    )

class SEN12MSSchema(BaseDatasetSchema):
    """
    Schema for configuring SEN12MS dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    imagery = fields.String(
        missing="s2", description="Consider s1, s2 or both s1 and s2 images", example="s1/s2/all"
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B02","B03","B04"]
    )
    bands_s1 = fields.List(
        fields.String(), missing=None, description="Select which S1 bands to use with a list of band names", example=["VV","VH"]
    )

class PASTISHDSchema(BaseDatasetSchema):
    """
    Schema for configuring PASTIS-HD dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    imagery = fields.String(
        missing="SPOT", description="read S1A, S1D, S2 and SPOT imagery or SPOT imagery only", example="all/SPOT"
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B01","B02","B03"]
    )
    bands_s1 = fields.List(
        fields.String(), missing=None, description="Select which s1 bands to use with a list of band names", example=["VV","VH"]
    )

class SatlasPretrainSemanticSegmentationSchema(BaseDatasetSchema):
    """
    Schema for configuring SatlasPretrain dataset for the semantic segmentation task.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    tile_list = fields.String(
        missing=None, description="Txt file with the list of tiles and their corresponding s2a, s2b, naip and label files", example="./data/tile_list.txt"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )

class SSL4EOS12MLSchema(BaseDatasetSchema):
    """
    Schema for configuring SSL4EO-S12-ML dataset.
    """
    label_file = fields.String(
        missing=None, description="json file on disk containing the labels", example="./data/labels.json"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk containing the train/val/test split", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    imagery = fields.String(
        missing="S2 L2A", description="Choose which imagery source to read", example="S2 L1C/S2 L2A/S1/all"
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B01","B02","B03"]
    )
    bands_s1 = fields.List(
        fields.String(), missing=None, description="Select which s1 bands to use with a list of band names", example=["VV","VH"]
    )

class DFC21DSESchema(BaseDatasetSchema):
    """
    Schema for configuring DFC21-DSE datasets.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    time_series = fields.Boolean(
        missing=False, description="Consider the time series or a single time step", example="True"
    )
    imagery = fields.String(
        missing="s2", description="Read S1, S2, Landsat 8 or VIIRS images", example="s1/s2/l8/viirs"
    )

class WHUOPTSARSchema(BaseDatasetSchema):
    """
    Schema for configuring WHU-OPT-SAR dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    imagery = fields.String(
        missing="s2", description="Consider s1, s2 or both s1 and s2 images", example="s1/s2/s1 and s2"
    )

class MultilabelChangeDetectionSchema(BaseDatasetSchema):
    """
    Schema for configuring Change Detection datasets with multilabel masks.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )
    task_type = fields.String(
        missing="multilabel", description="binary or multilabel", example="binary"
    )

class BandonSchema(BaseDatasetSchema):
    """
    Schema for configuring Bandon dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )
    split = fields.String(
        missing="original", description="original or new 60/20/20 split", example="new"
    )

class MSBCSchema(BaseDatasetSchema):
    """
    Schema for configuring MSBC dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )
    imagery = fields.String(
        missing="s2", description="Consider only one of s1, s2, aerial or all sources", example="s1/s2/aerial/all"
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B01","B02","B03"]
    )
    bands_s1 = fields.List(
        fields.String(), missing=None, description="Select which S1 bands to use with a list of band names", example=["VV","VH"]
    )

class SMARSSchema(BaseDatasetSchema):
    """
    Schema for configuring the SMARS dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )
    imagery = fields.String(
        missing="rgb", description="Consider only RGB, DEM or both sources", example="rgb/dem/all"
    )
    task_type = fields.String(
        missing="multilabel", description="binary or multilabel", example="binary"
    )

class TallosSchema(BaseDatasetSchema):
    """
    Schema for configuring the TalloS dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )
    annotations = fields.String(
        missing="/", description="Annotation file on disk", example="./data/annotations.csv",
    )
    task = fields.String(
        missing=None, description="Training, validation or testing task", example="val",
    )
    selection = fields.String(
        missing="rgb", description="Select 3 bands RGB images, specific bands or all bands", example="rgb/bands/all",
    )
    imagery = fields.String(
        missing="sentinel2", description="Select S2, DEM or all sources", example="dem/sentinel2/all",
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B01","B02","B03"]
    )

class Sentinel2CloudCoverSchema(BaseDatasetSchema):
    """
    Schema for configuring Sentinel-2 Cloud Cover dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels, specific channels or all channels", example="all/rgb/bands"
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B02","B03","B04"]
    )

class AIRPolSARSegSchema(BaseDatasetSchema):
    """
    Schema for configuring AIR-PolSAR-Seg dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels, specific channels or all channels", example="all/rgb/bands"
    )
    bands_gf3 = fields.List(
        fields.String(), missing=None, description="Select which GaoFen-3 bands to use with a list of band names", example=["HH","HV","VV"]
    )

class AgriFieldNetSchema(BaseDatasetSchema):
    """
    Schema for configuring AgriFieldNet dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels, specific channels or all channels", example="all/rgb/bands"
    )
    bands_s2 = fields.List(
        fields.String(), missing=None, description="Select which S2 bands to use with a list of band names", example=["B02","B03","B04"]
    )

class OmbriaSchema(BaseDatasetSchema):
    """
    Schema for configuring Ombria dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or all original channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )
    imagery = fields.String(
        missing="S2", description="Consider only one of s1, s2 or all sources", example="S1/S2/all"
    )

class RapidAI4EOSchema(BaseDatasetSchema):
    """
    Schema for configuring the RapidAI4EO dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    csv_file = fields.String(
        missing=None, description="CSV path on disk", example="./data/CESBIO/train.csv"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    date = fields.String(
        missing=None, description="Visualise images for a particular date", example="1"
    )  
    imagery = fields.String(
        missing="S2", description="Select which imagery to use: 'S2' for Sentinel-2, 'PF' for Planet Fusion", example="S2"
    )
    task = fields.String(
        missing=None, description="Select which split to use, train, val or test", example="train"
    )

class WHUOHSSchema(BaseDatasetSchema):
    """
    Schema for configuring WHU-OHS dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels, specific channels or all channels", example="all/rgb/bands"
    )
    bands_ohs = fields.List(
        fields.String(), missing=None, description="Select which OHS bands to use with a list of band names", example=["B02","B03","B04"]
    )

class ArtifivePotsdamSchema(BaseDatasetSchema):
    """
    Schema for configuring the Artifive-Potsdam dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    annotation_dir = fields.String(
        missing='/', description="json COCO files path on disk", example="./data/annotations/"
    )
    csv_file = fields.String(
        missing=None, description="CSV path on disk", example="./data/CESBIO/train.csv"
    )
    data_selection = fields.String(
        missing="labeled", description="Read all images or only those which contain at least one label", example="all/labeled"
    )
    imagery = fields.String(
        missing="all", description="Select which imagery to use: real, artificial or all", example="real/artificial/all"
    )

class RarePlanesSchema(BaseDatasetSchema):
    """
    Schema for configuring the RarePlanes dataset.
    """
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    annotation_dir = fields.String(
        missing='/', description="annotation files path on disk", example="./data/annotations/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    task = fields.String(
        missing=None, description="training, validation or test task", example="train"
    )
    imagery = fields.String(
        missing="real", description="Select which imagery to use: real, synthetic or all", example="real/synthetic/all"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )

class GeoNRWSchema(BaseDatasetSchema):
    """
    Schema for configuring GeoNRW dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    imagery = fields.String(
        missing="aerial", description="Select aerial, DEM or all sources", example="dem/aerial/all",
    )

class xView2Schema(BaseDatasetSchema):
    """
    Schema for configuring xView2 dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    classification = fields.String(
        missing="binary", description="Classification schema", example="multilabel/binary"
    )

class Sen1Floods11Schema(BaseDatasetSchema):
    """
    Schema for configuring Sen1Floods11 dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    source_dir = fields.String(
        missing="/", description="Source images path on disk", example="./data/CESBIO/"
    )
    annot_dir = fields.String(
        missing="/", description="Annotations path on disk", example="./data/CESBIO/"
    )
    imagery = fields.String(
        missing=None, description="Imagery source", example="S1/S2"
    )
    annotations = fields.String(
        missing=None, description="Annotations type", example="weakly labeled/hand labeled"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )

class DynamicEarthNetSchema(BaseDatasetSchema):
    """
    Schema for configuring DynamicEarthNet dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )
    imagery = fields.String(
        missing="s2", description="S1, S2, PlanetScope imagery or all sources", example="s1/s2/planet/all"
    )
    timeseries= fields.Boolean(
        missing=False, description="Consider the timeseries or a single timestamp", example="True/False"
    )

class BioMasstersSchema(BaseDatasetSchema):
    """
    Schema for configuring BioMassters dataset.
    """
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/CESBIO/"
    )
    annotation_dir = fields.String(
        missing="/", description="Annotations path on disk", example="./data/CESBIO/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or more than 3 channels", example="all/rgb"
    )
    mode = fields.String(
        missing=None, description="train/val or test task", example="train"
    )
    timeseries = fields.Boolean(
        missing=False, description="Consider the time series or a single time stamp", example="True/False"
    )
    imagery = fields.String(
        missing="S2", description="Consider S1, S2 or both sources", example="S1/S2/all"
    )