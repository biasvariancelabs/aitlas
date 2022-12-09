from marshmallow import Schema, fields, validate


class BaseDatasetSchema(Schema):
    batch_size = fields.Int(missing=64, description="Batch size", example=64)
    shuffle = fields.Bool(
        missing=True, description="Should shuffle dataset", example=False
    )
    num_workers = fields.Int(missing=4, description="Number of workers", example=4)
    pin_memory = fields.Bool(
        missing=False, description="Whether to use page-locked memory"
    )
    transforms = fields.List(
        fields.String, missing=None, description="Classes to run transformations.",
    )
    target_transforms = fields.List(
        fields.String, missing=None, description="Classes to run transformations.",
    )
    joint_transforms = fields.List(
        fields.String, missing=None, description="Classes to run transformations.",
    )
    labels = fields.List(
        fields.String, missing=None, description="Labels for the dataset",
    )


class BaseModelSchema(Schema):
    num_classes = fields.Int(missing=2, description="Number of classes", example=2)
    use_cuda = fields.Bool(missing=True, description="Whether to use CUDA if possible")
    metrics = fields.List(
        fields.String,
        missing=["f1_score"],
        description="Metrics you want to calculate",
        example=["accuracy", "precision", "iou"],
        validate=validate.ContainsOnly(
            ["accuracy", "precision", "recall", "f1_score", "iou", "kappa", "map"]
        ),
    )
    weights = fields.List(
        fields.Float,
        missing=None,
        description="Classes weights you want to apply for the loss",
        example=[1.0, 2.3, 1.0],
    )
    rank = fields.Integer(required=False, missing=0)
    use_ddp = fields.Boolean(
        required=False, missing=False, description="Turn on distributed data processing"
    )


class BaseClassifierSchema(BaseModelSchema):
    learning_rate = fields.Float(
        missing=0.01, description="Learning rate used in training.", example=0.01
    )
    weight_decay = fields.Float(
        missing=0.0, description="Learning rate used in training.", example=0.01
    )
    pretrained = fields.Bool(
        missing=True, description="Whether to use a pretrained network or not."
    )
    local_model_path = fields.String(
        missing=None, description="Local path of the pre-trained model",
    )
    threshold = fields.Float(
        missing=0.5, description="Prediction threshold if needed", example=0.5
    )
    freeze = fields.Bool(
        missing=False,
        description="Whether to freeze all the layers except for the classifier layer(s).",
    )


class BaseSegmentationClassifierSchema(BaseClassifierSchema):
    metrics = fields.List(
        fields.String,
        missing=["iou", "f1_score", "accuracy"],
        description="Classes of metrics you want to calculate",
        example=["accuracy", "precision", "recall", "f1_score", "iou"],
    )


class BaseObjectDetectionSchema(BaseClassifierSchema):
    metrics = fields.List(
        fields.String,
        missing=["map"],
        description="Classes of metrics you want to calculate",
        example=["accuracy", "precision", "recall", "f1_score", "iou"],
    )
    step_size = fields.Integer(missing=15, description="Step size for LR scheduler.",)
    gamma = fields.Float(
        missing=0.1, description="Gamma (multiplier) for LR scheduler.",
    )


class BaseTransformsSchema(Schema):
    pass
