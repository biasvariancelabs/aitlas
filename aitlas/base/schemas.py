from marshmallow import Schema, fields


class BaseDatasetSchema(Schema):
    batch_size = fields.Int(missing=64, description="Batch size", example=64)
    shuffle = fields.Bool(
        missing=True, description="Should shuffle dataset", example=False
    )
    num_workers = fields.Int(missing=4, description="Number of workers", example=4)
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


class BaseClassifierSchema(Schema):
    num_classes = fields.Int(missing=2, description="Number of classes", example=2)
    learning_rate = fields.Float(
        missing=None, description="Learning rate used in training.", example=0.01
    )
    use_cuda = fields.Bool(missing=True, description="Whether to use CUDA if possible")
    pretrained = fields.Bool(
        missing=True, description="Whether to use a pretrained network or not."
    )
    threshold = fields.Float(
        missing=0.5, description="Prediction threshold if needed", example=0.5
    )
    extract_feature_only = fields.Bool(
        missing=False,
        description="Whether to use the network without the classification layer.",
    )


class BaseTransformsSchema(Schema):
    pass
