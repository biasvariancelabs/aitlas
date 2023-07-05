from marshmallow import Schema, fields, validate


class BaseDatasetSchema(Schema):

    """
    Schema for configuring a base dataset.

    :param batch_size: Batch size for the dataset. Default is 64.
    :type batch_size: int, optional

    :param shuffle: Flag indicating whether to shuffle the dataset. Default is True.
    :type shuffle: bool, optional

    :param num_workers: Number of workers to use for data loading. Default is 4.
    :type num_workers: int, optional

    :param pin_memory: Flag indicating whether to use page-locked memory. Default is False.
    :type pin_memory: bool, optional

    :param transforms: Classes to run transformations over the input data.
    :type transforms: List[str], optional

    :param target_transforms: Classes to run transformations over the target data.
    :type target_transforms: List[str], optional

    :param joint_transforms: Classes to run transformations over the input and target data.
    :type joint_transforms: List[str], optional

    :param labels: Labels for the dataset.
    :type labels: List[str], optional
    """

    batch_size = fields.Int(missing=64, description="Batch size", example=64)
    shuffle = fields.Bool(
        missing=True, description="Should shuffle dataset", example=False
    )
    num_workers = fields.Int(missing=4, description="Number of workers", example=4)
    pin_memory = fields.Bool(
        missing=False, description="Whether to use page-locked memory"
    )
    transforms = fields.List(
        fields.String,
        missing=None,
        description="Classes to run transformations over the input data.",
    )
    target_transforms = fields.List(
        fields.String,
        missing=None,
        description="Classes to run transformations over the target data.",
    )
    joint_transforms = fields.List(
        fields.String,
        missing=None,
        description="Classes to run transformations over the input and target data.",
    )
    labels = fields.List(
        fields.String,
        missing=None,
        description="Labels for the dataset",
    )


class BaseModelSchema(Schema):
    """
    Schema for configuring a base model.

    :param num_classes: Number of classes for the model. Default is 2.
    :type num_classes: int, optional

    :param use_cuda: Flag indicating whether to use CUDA if available. Default is True.
    :type use_cuda: bool, optional

    :param metrics: Metrics to calculate during training and evaluation. Default is ['f1_score'].
    :type metrics: List[str], optional

    :param weights: Class weights to apply for the loss function. Default is None.
    :type weights: List[float], optional

    :param rank: Rank value for distributed data processing. Default is 0.
    :type rank: int, optional

    :param use_ddp: Flag indicating whether to turn on distributed data processing. Default is False.
    :type use_ddp: bool, optional
    """

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
    """
    Schema for configuring a base classifier.

    :param learning_rate: Learning rate used in training. Default is 0.01.
    :type learning_rate: float, optional

    :param weight_decay: Weight decay used in training. Default is 0.0.
    :type weight_decay: float, optional

    :param pretrained: Flag indicating whether to use a pretrained model. Default is True.
    :type pretrained: bool, optional

    :param local_model_path: Local path of the pretrained model. Default is None.
    :type local_model_path: str, optional

    :param threshold: Prediction threshold if needed. Default is 0.5.
    :type threshold: float, optional

    :param freeze: Flag indicating whether to freeze all layers except for the classifier layer(s). Default is False.
    :type freeze: bool, optional
    """

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
        missing=None,
        description="Local path of the pre-trained model",
    )
    threshold = fields.Float(
        missing=0.5, description="Prediction threshold if needed", example=0.5
    )
    freeze = fields.Bool(
        missing=False,
        description="Whether to freeze all the layers except for the classifier layer(s).",
    )


class BaseSegmentationClassifierSchema(BaseClassifierSchema):
    """
    Schema for configuring a base segmentation classifier.

    :param metrics: Classes of metrics you want to calculate during training and evaluation.
        Default is ['iou', 'f1_score', 'accuracy'].
    :type metrics: List[str], optional
    """

    metrics = fields.List(
        fields.String,
        missing=["iou", "f1_score", "accuracy"],
        description="Classes of metrics you want to calculate",
        example=["accuracy", "precision", "recall", "f1_score", "iou"],
    )


class BaseObjectDetectionSchema(BaseClassifierSchema):
    """
    Schema for configuring a base object detection model.

    :param metrics: Classes of metrics you want to calculate during training and evaluation.
        Default is ['map'].
    :type metrics: List[str], optional

    :param step_size: Step size for the learning rate scheduler. Default is 15.
    :type step_size: int, optional

    :param gamma: Gamma (multiplier) for the learning rate scheduler. Default is 0.1.
    :type gamma: float, optional
    """

    metrics = fields.List(
        fields.String,
        missing=["map"],
        description="Classes of metrics you want to calculate",
        example=["accuracy", "precision", "recall", "f1_score", "iou"],
    )
    step_size = fields.Integer(
        missing=15,
        description="Step size for LR scheduler.",
    )
    gamma = fields.Float(
        missing=0.1,
        description="Gamma (multiplier) for LR scheduler.",
    )


class BaseTransformsSchema(Schema):
    pass
