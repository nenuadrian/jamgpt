import os

import torch
import pyarrow.parquet as pq
from collections import deque

from jamgpt.common import get_dist_info


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


def tokenizing_distributed_data_loader(
    B, T, split, data_dir, tokenizer, tokenizer_threads=4, tokenizer_batch_size=128
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

    def document_batches():
        while True:
            for texts in parquets_iter_batched(
                split=split,
                data_dir=data_dir,
                start=ddp_rank,
                step=ddp_world_size,
            ):
                for i in range(0, len(texts), tokenizer_batch_size):
                    yield texts[i : i + tokenizer_batch_size]

    batches = document_batches()

    batch_index = 0
    while True:
        while len(token_buffer) < needed_tokens:
            text_batch = next(batches)
            token_batches = tokenizer.encode_batch(
                text_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for token_ids in token_batches:
                token_buffer.extend(token_ids)
            batch_index += 1

        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        input_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]

        inputs = input_cpu.view(B, T).to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )
        targets = targets_cpu.view(B, T).to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )
        yield inputs, targets
