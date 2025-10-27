import os
import time
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading


import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT",
        help="Output directory for parquet shards",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Dataset path",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="sample-100BT", help="Dataset name/config"
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split"
    )
    parser.add_argument(
        "--chars_per_shard",
        type=int,
        default=250_000_000,
        help="Number of characters per shard",
    )
    parser.add_argument(
        "--row_group_size",
        type=int,
        default=1024,
        help="Row group size for parquet files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Maximum number of documents to download (None for all)",
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Maximum number of shards to create (None for all)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode to avoid loading entire dataset into memory",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for parallel dataset loading (non-streaming only)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel downloads (streaming mode)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set cache directory to be within output directory
    cache_dir = os.path.join(args.output_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["HF_MODULES_CACHE"] = os.path.join(cache_dir, "modules")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")

    from datasets import load_dataset

    DATASET_KWARGS = {
        "path": args.dataset_path,
        "split": args.dataset_split,
        "name": args.dataset_name,
        "cache_dir": cache_dir,
        "streaming": args.streaming,
    }

    # Add num_proc for non-streaming mode
    if not args.streaming and args.num_proc is not None:
        DATASET_KWARGS["num_proc"] = args.num_proc

    print(f"Using cache directory at {cache_dir}")
    chars_per_shard = args.chars_per_shard
    row_group_size = args.row_group_size

    print("Loading dataset...")
    ds = load_dataset(**DATASET_KWARGS)

    # For streaming datasets, we can't get exact length or shuffle globally
    # For non-streaming, we can still shuffle if memory allows
    if not args.streaming:
        print("Shuffling dataset...")
        ds = ds.shuffle(seed=args.seed)
        ndocs = len(ds)
        if args.max_docs is not None:
            ndocs = min(ndocs, args.max_docs)
            ds = ds.select(range(ndocs))
            print(f"Limited to {ndocs} documents")
        print(f"Number of documents: {ndocs}")
    else:
        print(f"Using streaming mode (buffer-based shuffling)")
        # For streaming, shuffle at the buffer level
        ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
        ndocs = args.max_docs if args.max_docs is not None else None

    os.makedirs(args.output_dir, exist_ok=True)

    shard_docs = []
    shard_index = 0
    shard_chars = 0
    docs_processed = 0
    time_spent = 0
    time_start = time.time()

    # Thread-safe lock for writing shards (not needed in current implementation but kept for safety)
    write_lock = threading.Lock()

    # Create progress bar
    pbar = tqdm(total=ndocs, desc="Processing documents", unit="docs")

    for doc in ds:
        # Check if we've reached max shards limit
        if args.max_shards is not None and shard_index >= args.max_shards:
            print(
                f"\nReached maximum number of shards ({args.max_shards}), stopping..."
            )
            break

        # Check if we've reached max docs limit (for streaming)
        if args.max_docs is not None and docs_processed >= args.max_docs:
            print(
                f"\nReached maximum number of documents ({args.max_docs}), stopping..."
            )
            break

        # Only append the text, not the full document
        text = doc["text"]
        shard_docs.append(text)
        shard_chars += len(text)
        docs_processed += 1
        docs_multiple_of_row_group = docs_processed % row_group_size == 0

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix(
            {
                "shards": shard_index,
                "chars": f"{shard_chars:,}",
                "MB": f"{shard_chars / 1_000_000:.1f}",
            }
        )

        if shard_chars >= chars_per_shard and docs_multiple_of_row_group:
            table = pa.Table.from_pydict({"text": shard_docs})

            pq.write_table(
                table,
                os.path.join(args.output_dir, f"shard_{shard_index:05d}.parquet"),
                row_group_size=row_group_size,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )

            shard_index += 1
            # Clear the list to free memory
            shard_docs.clear()
            shard_chars = 0

            time_end = time.time()
            time_spent += time_end - time_start
            time_start = time_end

            # Update progress bar with shard completion info
            pbar.set_description(f"Processing docs (shard {shard_index} saved)")

    pbar.close()

    # Save any remaining documents as final shard
    if shard_docs and (args.max_shards is None or shard_index < args.max_shards):
        print(f"\nSaving final shard with {len(shard_docs)} documents...")
        table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            table,
            os.path.join(args.output_dir, f"shard_{shard_index:05d}.parquet"),
            row_group_size=row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        shard_index += 1

    print(
        f"\n✓ Complete: Created {shard_index} shard(s) from {docs_processed:,} documents"
    )
    print(f"  Total time: {time_spent:.2f} seconds")
    print(f"  Average: {time_spent/docs_processed:.3f} sec/doc")

    # Clean up default HuggingFace cache if requested
    if os.path.exists(cache_dir):
        print(f"\nCleaning up cache at {cache_dir}")
        shutil.rmtree(cache_dir)
        print("✓ Cache cleaned up successfully")
