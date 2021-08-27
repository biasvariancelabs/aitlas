import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from .base import Config, RunConfig
from .utils import get_class


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size, task):
    setup(rank, world_size)

    task.run()

    cleanup()


def main(config_file):
    """Main entry function to the toolbox"""

    with open(config_file) as file:
        config = json.load(file)

    # load configuration
    config = Config(RunConfig().load(config))

    # check if there are multiple GPUs
    world_size = torch.cuda.device_count()

    # load model, if specified
    model = None
    if config.model:
        model_cls = get_class(config.model.classname)
        model = model_cls(config.model.config)
        model.prepare()

    # load task
    task_cls = get_class(config.task.classname)
    task = task_cls(model, config.task.config)

    # run task
    mp.spawn(run, args=(world_size, task,), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs a workflow described in the configuration file."
    )

    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    main(config_file=args.config)
