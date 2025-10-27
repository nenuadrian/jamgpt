import os
import argparse
import glob
from typing import Iterator
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import json
from multiprocessing import Pool, cpu_count, get_context

# Set environment variable before any tokenizer operations
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT",
        help="Path to directory containing parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer_output",
        help="Output directory for trained tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Vocabulary size for the tokenizer",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of parquet files to use (None for all)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N files",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel workers for reading files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for reading parquet files",
    )
    return parser.parse_args()


def save_checkpoint(output_dir: str, processed_files: list, file_index: int):
    """Save checkpoint with processed files list only."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "processed_files": processed_files,
        "file_index": file_index,
    }

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint(output_dir: str):
    """Load checkpoint if exists."""
    checkpoint_path = os.path.join(output_dir, "checkpoints", "checkpoint.json")

    if not os.path.exists(checkpoint_path):
        return None

    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    return checkpoint


def process_parquet_file(args_tuple):
    """Worker function to process a single parquet file and return texts."""
    file_path, batch_size = args_tuple
    texts = []

    try:
        parquet_file = pq.ParquetFile(file_path)

        # Stream batches instead of loading entire file
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_texts = batch.column("text").to_pylist()

            # Filter valid texts
            texts.extend([t for t in batch_texts if t and isinstance(t, str)])

        return file_path, texts, None
    except Exception as e:
        return file_path, [], str(e)


def text_iterator_from_parquet(
    dataset_path: str,
    max_files: int = None,
    output_dir: str = None,
    checkpoint_interval: int = 1,
    resume_from_checkpoint: bool = False,
    num_workers: int = 4,
    batch_size: int = 10000,
) -> Iterator[str]:
    """Yield text from parquet files with parallel processing and streaming."""
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    # Load checkpoint if resuming
    processed_files_set = set()

    if resume_from_checkpoint and output_dir:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            processed_files_set = set(checkpoint["processed_files"])

    # Filter out already processed files
    parquet_files = [f for f in parquet_files if f not in processed_files_set]
    total_files = len(parquet_files)
    
    if total_files == 0:
        return

    processed_files = list(processed_files_set)

    # Process files in larger chunks for better throughput
    chunk_size = num_workers * 4

    for chunk_start in range(0, total_files, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_files)
        chunk = parquet_files[chunk_start:chunk_end]

        # Prepare arguments for parallel processing
        worker_args = [(file_path, batch_size) for file_path in chunk]

        # Use imap_unordered for streaming results as they complete
        with get_context("spawn").Pool(processes=num_workers) as pool:
            for file_path, texts, error in pool.imap_unordered(process_parquet_file, worker_args):
                if error:
                    if output_dir:
                        save_checkpoint(output_dir, processed_files, len(processed_files))
                    raise RuntimeError(f"Failed to process {file_path}: {error}")

                file_num = len(processed_files) + 1
                print(f"[{file_num}/{total_files}] {os.path.basename(file_path)}")

                # Yield all texts from this file
                for text in texts:
                    yield text

                processed_files.append(file_path)

                # Save checkpoint periodically
                if output_dir and file_num % checkpoint_interval == 0:
                    save_checkpoint(output_dir, processed_files, file_num)

    # Save final checkpoint
    if output_dir:
        save_checkpoint(output_dir, processed_files, total_files)


def train_bpe_tokenizer(args):
    """Train a BPE tokenizer similar to GPT-2."""

    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer (GPT-2 style: split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Set post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Define special tokens
    special_tokens = ["<|endoftext|>", "<|pad|>"]

    # Initialize trainer
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=False,
    )

    print(f"Training tokenizer (vocab_size={args.vocab_size})")
    print(f"Dataset: {args.dataset_path}\n")

    # Train from iterator
    try:
        tokenizer.train_from_iterator(
            text_iterator_from_parquet(
                args.dataset_path,
                args.max_files,
                output_dir=args.output_dir,
                checkpoint_interval=args.checkpoint_interval,
                resume_from_checkpoint=args.resume_from_checkpoint,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            ),
            trainer=trainer,
        )
    except Exception as e:
        print(f"\nError: {e}")
        print("Resume with --resume_from_checkpoint")
        raise

    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    print(f"\nTokenizer saved to {args.output_dir}")

    # Clean up checkpoints after successful training
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)

    return tokenizer


def test_tokenizer(tokenizer):
    """Quick test of the trained tokenizer."""
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
    ]

    print("\nTesting tokenizer:")
    for text in test_texts:
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids)
        print(f"\nOriginal: {text}")
        print(f"Tokens: {encoding.tokens}")
        print(f"IDs: {encoding.ids}")
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    args = parse_args()
    tokenizer = train_bpe_tokenizer(args)
    test_tokenizer(tokenizer)
    """Quick test of the trained tokenizer."""
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
    ]

    print("\nTesting tokenizer:")
    for text in test_texts:
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids)
        print(f"\nOriginal: {text}")
        print(f"Tokens: {encoding.tokens}")
        print(f"IDs: {encoding.ids}")
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    args = parse_args()
    tokenizer = train_bpe_tokenizer(args)
    test_tokenizer(tokenizer)
