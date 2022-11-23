import argparse
import json

import ignite.distributed as idist
import torch

from .base import Config, RunConfig
from .utils import get_class


torch.manual_seed(1)


def run(rank, config):
    # load model, if specified
    model = None
    if config.model:
        # pass additional parameters to the model
        config.model.config.rank = rank
        config.model.config.use_ddp = config.use_ddp

        # initialize model
        model_cls = get_class(config.model.classname)
        model = model_cls(config.model.config)
        model.prepare()

    # load task
    task_cls = get_class(config.task.classname)
    task = task_cls(model, config.task.config)
    task.run()


def main(config_file):
    """Main entry function to the toolbox"""

    with open(config_file) as file:
        config = json.load(file)

    # load configuration
    config = Config(RunConfig().load(config))

    # run task
    if config.use_ddp:
        # check if there are multiple GPUs
        world_size = torch.cuda.device_count()

        # spawn processes
        with idist.Parallel(backend="gloo", nproc_per_node=world_size) as parallel:
            parallel.run(run, config)
    else:
        run(0, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs a workflow described in the configuration file."
    )

    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    main(config_file=args.config)
