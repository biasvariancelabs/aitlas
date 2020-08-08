from marshmallow import Schema, fields


class BaseDatasetSchema(Schema):
    batch_size = fields.Int(missing=4, description="Batch size", example=4)
    shuffle = fields.Bool(
        missing=True, description="Should shuffle dataset", example=False
    )
    num_workers = fields.Int(missing=2, description="Number of workers", example=2)
