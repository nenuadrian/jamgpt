import os
import logging
import torch
import torch.distributed as dist
import pyarrow.parquet as pq


def list_parquet_files(data_dir=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
            and not f.endswith(".tmp")
            and not f.startswith(".")
        ]
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, data_dir: str, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        print(f"Loading parquet file: {filepath}")
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


def setup_default_logging():
    handler = logging.StreamHandler()
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir():
    # co-locat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("JAMGPT_BASE_DIR"):
        jamgpt_dir = os.environ.get("JAMGPT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        jamgpt_dir = os.path.join(cache_dir, "jamgpt")
    os.makedirs(jamgpt_dir, exist_ok=True)
    return jamgpt_dir


def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get("RANK", -1)) != -1


def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def compute_init():
    """Basic initialization that we keep doing over and over, so make common."""

    # CUDA is currently required
    assert torch.cuda.is_available(), "CUDA is needed for a distributed run atm"

    # Reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Precision
    torch.set_float32_matmul_precision("high")  # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda")

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""

    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass
