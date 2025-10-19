import os
import time
import argparse
import shutil


import pyarrow.parquet as pq
import pyarrow as pa


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
        "streaming": True,
    }
    print(f"Using cache directory at {cache_dir}")
    chars_per_shard = args.chars_per_shard
    row_group_size = args.row_group_size

    ds = load_dataset(**DATASET_KWARGS)

    ds = ds.shuffle(seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    shard_docs = []
    shard_index = 0
    shard_chars = 0
    docs_processed = 0
    time_spent = 0
    time_start = time.time()

    for doc in ds:
        shard_chars += len(doc["text"])
        docs_processed += 1
        shard_docs.append(doc["text"])
        if shard_chars >= chars_per_shard and docs_processed % row_group_size == 0:
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
            shard_docs = []
            shard_chars = 0

            time_end = time.time()
            time_spent += time_end - time_start
            time_start = time_end
            print(
                f"Processed {docs_processed} documents "
                f"time spent: {time_spent:.2f} seconds"
            )

    # Clean up default HuggingFace cache if requested
    if os.path.exists(cache_dir):
        print(f"Cleaning up cache at {cache_dir}")
        shutil.rmtree(cache_dir)
        print("Cache cleaned up successfully")
