from marshmallow import Schema, fields


class BaseClassifierSchema(Schema):
    num_classes = fields.Int(required=True, description="Number of classes", example=2)
    learning_rate = fields.Float(
        missing=None, description="Learning rate used in training.", example=0.01
    )
    class_weights = fields.Dict(
        missing=None,
        description="Dictionary mapping class id with weight. "
        "If key for some labels is not specified, 1 is used.",
    )
