from marshmallow import fields, validate

from aitlas.base.schemas import BaseDatasetSchema

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