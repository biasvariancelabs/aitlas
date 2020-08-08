import argparse
import json

from marshmallow import Schema, fields, post_load

from .base.config import Config, ObjectConfig
from .utils import get_class


class RunConfig(Schema):
    """Top level configuration schema"""

    model = fields.Nested(
        ObjectConfig, required=True, description="Model configuration"
    )
    dataset = fields.Nested(
        ObjectConfig, required=True, description="Dataset configuration"
    )
    task = fields.Nested(ObjectConfig, required=True, description="Task configuration")


def main(config_file):
    """Main entry function to the toolbox"""

    with open(config_file) as file:
        config = json.load(file)

    # load configuration
    config = Config(RunConfig().load(config))

    # load model
    model_cls = get_class(config.model.classname)
    model = model_cls(config.model.config)

    # load dataset
    dataset_cls = get_class(config.dataset.classname)
    dataset = dataset_cls(config.dataset.config)

    # load task
    task_cls = get_class(config.task.classname)
    task = task_cls(model, dataset, config.task.config)

    # run task
    task.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs a workflow described in the configuration file."
    )

    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    main(config_file=args.config)
