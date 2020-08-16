import argparse
import json

from .base import Config, RunConfig
from .utils import get_class


def main(config_file):
    """Main entry function to the toolbox"""

    with open(config_file) as file:
        config = json.load(file)

    # load configuration
    config = Config(RunConfig().load(config))

    # load model, if specified
    model = None
    if config.model:
        model_cls = get_class(config.model.classname)
        model = model_cls(config.model.config)

    # load dataset
    dataset_cls = get_class(config.dataset.classname)
    dataset = dataset_cls(config.dataset.config)
    # prepare the dataset
    dataset.prepare()

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
