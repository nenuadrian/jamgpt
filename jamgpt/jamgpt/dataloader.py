from collections import deque

import torch

from jamgptcommon import get_dist_info
from jamgpttokenizer import get_tokenizer


import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from jamgptcommon import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
MAX_SHARD = 1822  # the last datashard is shard_01822.parquet
index_to_filename = (
    lambda index: f"shard_{index:05d}.parquet"
)  # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


def list_parquet_files(data_dir=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


# -----------------------------------------------------------------------------
def download_single_file(index):
    """Downloads a single file index, with some backoff"""

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def tokenizing_distributed_data_loader(
    B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"
):
    """Stream pretraining text from parquet files, tokenize, yield training batches."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque()  # we stream tokens on the right and pop from the left

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(
                split=split, start=ddp_rank, step=ddp_world_size
            ):
                # for the tokenizer we might want to go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size]

    batches = document_batches()

    batch_index = 0
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(
                doc_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=True)
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        targets = targets_cpu.view(B, T).to(
            device=device, dtype=torch.int64, non_blocking=True
        )
        yield inputs, targets
