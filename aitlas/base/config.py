from abc import ABC

from marshmallow import Schema, fields
from munch import Munch


class Config(Munch):
    """ Config object used for automatic object creation from a dict.
    """

    def __init__(self, config):
        def convert(obj):
            """ Recursively convert a dict to Munch. (there is a Munch.from_dict method, but it's not python3 compatible)
            """
            if isinstance(obj, list):
                return [convert(element) for element in obj]
            if isinstance(obj, dict):
                return Munch({k: convert(v) for k, v in obj.items()})
            return obj

        config = convert(config)

        super().__init__(config)


class ObjectConfig(Schema):
    classname = fields.String(required=True, description="Class to instantiate.")
    config = fields.Dict(
        required=True, descripton="Configuration used for instantiation of the class."
    )


class RunConfig(Schema):
    """Top level configuration schema"""

    model = fields.Nested(ObjectConfig, missing=None, description="Model configuration")
    task = fields.Nested(ObjectConfig, required=True, description="Task configuration")
    use_ddp = fields.Boolean(
        required=False, missing=False, description="Turn on distributed data processing"
    )


class Configurable(ABC):
    """ Base class for all configurable objects.
    """

    schema = None  # you need specify the schema of the class

    def __init__(self, config):
        if not self.schema:
            raise ValueError(f"You are missing a schema for {self.__class__}")
        self.config = Config(self.schema().load(config))
