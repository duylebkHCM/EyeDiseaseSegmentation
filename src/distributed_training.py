import torch
from torch.utils.data import TensorDataset
from catalyst.dl import SupervisedRunner
from typing import Callable
from catalyst.utils.distributed import (
    get_distributed_env,
    get_distributed_params,
)
import warnings
import subprocess
import sys
import time

def distributed_cmd_run(
    worker_fn: Callable, distributed: bool = True, *args, **kwargs
) -> None:
    """
    Distributed run
    Args:
        worker_fn: worker fn to run in distributed mode
        distributed: distributed flag
        args: additional parameters for worker_fn
        kwargs: additional key-value parameters for worker_fn
    """
    distributed_params = get_distributed_params()
    local_rank = distributed_params["local_rank"]
    world_size = distributed_params["world_size"]

    if distributed and torch.distributed.is_initialized():
        warnings.warn(
            "Looks like you are trying to call distributed setup twice, "
            "switching to normal run for correct distributed training."
        )

    if (
        not distributed
        or torch.distributed.is_initialized()
        or world_size <= 1
    ):
        worker_fn(*args, **kwargs)
    elif local_rank is not None:
        torch.cuda.set_device(int(local_rank))

        torch.distributed.init_process_group(
            backend="gloo", init_method="env://"
        )
        worker_fn(*args, **kwargs)
    else:
        workers = []
        try:
            for local_rank in range(torch.cuda.device_count()):
                rank = distributed_params["start_rank"] + local_rank
                env = get_distributed_env(local_rank, rank, world_size)
                cmd = [sys.executable] + sys.argv.copy()
                workers.append(subprocess.Popen(cmd, env=env))
            for worker in workers:
                worker.wait()
        finally:
            for worker in workers:
                worker.kill()



def datasets_fn(num_features: int):
    X = torch.rand(int(1e4), num_features)
    y = torch.rand(X.shape[0])
    dataset = TensorDataset(X, y)
    return {"train": dataset, "valid": dataset}


def train():
    num_features = int(1e1)
    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    runner = SupervisedRunner()
    runner.train(
        model=model,
        datasets={
            "batch_size": 32,
            "num_workers": 20,
            "get_datasets_fn": datasets_fn,
            "num_features": num_features,
        },
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        logdir="./logs/example_3",
        num_epochs=8,
        verbose=True,
        distributed=False,
    )

# start = time.time()
# distributed_cmd_run(train)
# end = time.time()
# print(f'Total {(end-start)} seconds')

if __name__ == "__main__":
    start = time.time()

    train()
    end = time.time()
    print(f'Total {(end-start)} seconds')
