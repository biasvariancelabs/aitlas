from marshmallow import Schema, fields


class BaseDatasetSchema(Schema):
    batch_size = fields.Int(missing=64, description="Batch size", example=64)
    shuffle = fields.Bool(
        missing=True, description="Should shuffle dataset", example=False
    )
    num_workers = fields.Int(missing=4, description="Number of workers", example=4)
    pin_memory = fields.Bool(missing=False, description="Whether to use page-locked memory")
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


class BaseModelSchema(Schema):
    num_classes = fields.Int(missing=2, description="Number of classes", example=2)
    use_cuda = fields.Bool(missing=True, description="Whether to use CUDA if possible")
    metrics = fields.List(
        fields.String,
        missing=["aitlas.metrics.F1Score"],
        description="Classes of metrics you want to calculate",
        example=[
            "aitlas.metrics.PrecisionScore",
            "aitlas.metrics.AccuracyScore",
            "aitlas.metrics.F1Score",
        ],
    )


class BaseClassifierSchema(BaseModelSchema):
    learning_rate = fields.Float(
        missing=None, description="Learning rate used in training.", example=0.01
    )
    pretrained = fields.Bool(
        missing=True, description="Whether to use a pretrained network or not."
    )
    threshold = fields.Float(
        missing=0.5, description="Prediction threshold if needed", example=0.5
    )


class BaseSegmentationClassifierSchema(BaseClassifierSchema):
    metrics = fields.List(
        fields.String,
        missing=["aitlas.metrics.F1ScoreSample"],
        description="Classes of metrics you want to calculate",
        example=["aitlas.metrics.F1ScoreSample", "aitlas.metrics.Accuracy"],
    )


class BaseTransformsSchema(Schema):
    pass
