from marshmallow import Schema, fields


class TrainTaskSchema(Schema):
    num_epochs = fields.Int(
        required=True, description="Number of epochs used in training", example=50
    )
    iterations_per_epoch = fields.Int(
        required=True, description="Number of training steps per epoch", example=100
    )
    model_directory = fields.String(
        required=True,
        description="Directory of the model output",
        example="/tmp/model/",
    )
    save_steps = fields.Int(
        missing=100, description="Number of training steps between model checkpoints."
    )
