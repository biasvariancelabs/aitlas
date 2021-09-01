import argparse
import json
import os
import signal

import ignite.distributed as idist
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from .base import Config, RunConfig
from .utils import get_class


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["LOCAL_RANK"] = rank

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def signal_handler(signal, frame):
    cleanup()


# def run(rank, world_size, config):
def run(rank, config):
    # signal.signal(signal.SIGINT, signal_handler)
    # setup(rank, world_size)
    # print(f"Running basic DDP example on rank {rank}.")

    # load model, if specified
    model = None
    if config.model:
        config.model.config.rank = rank
        model_cls = get_class(config.model.classname)
        model = model_cls(config.model.config)
        model.prepare()

    # load task
    task_cls = get_class(config.task.classname)
    task = task_cls(model, config.task.config)
    task.run()

    # cleanup()


def main(config_file):
    """Main entry function to the toolbox"""

    with open(config_file) as file:
        config = json.load(file)

    # load configuration
    config = Config(RunConfig().load(config))

    # check if there are multiple GPUs
    world_size = 2  # torch.cuda.device_count()

    # run task
    if world_size > 1:
        # mp.spawn(run, args=(world_size, config,), nprocs=world_size, join=True)
        with idist.Parallel(backend="nccl", nproc_per_node=world_size) as parallel:
            parallel.run(run, config)
    else:
        run(0, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs a workflow described in the configuration file."
    )

    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    print("Do")
    main(config_file=args.config)
    print("Done")
