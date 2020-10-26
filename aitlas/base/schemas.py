from marshmallow import Schema, fields


class BaseDatasetSchema(Schema):
    batch_size = fields.Int(missing=64, description="Batch size", example=64)
    shuffle = fields.Bool(
        missing=True, description="Should shuffle dataset", example=False
    )
    num_workers = fields.Int(missing=4, description="Number of workers", example=4)


class BaseClassifierSchema(Schema):
    num_classes = fields.Int(required=True, description="Number of classes", example=2)
    learning_rate = fields.Float(
        missing=None, description="Learning rate used in training.", example=0.01
    )
    use_cuda = fields.Bool(missing=True, description="Whether to use CUDA if possible")
    pretrained = fields.Bool(
        missing=False, description="Whether to use a pretrained network or not."
    )
