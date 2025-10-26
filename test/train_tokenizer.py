import os
import argparse
import glob
from typing import Iterator
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import json
import pickle


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
    return parser.parse_args()


def save_checkpoint(
    output_dir: str, processed_files: list, texts_buffer: list, file_index: int
):
    """Save checkpoint with processed files list and current text buffer."""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "processed_files": processed_files,
        "texts_buffer": texts_buffer,
        "file_index": file_index,
    }

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    # Also save a readable JSON for debugging
    json_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "processed_files": processed_files,
                "file_index": file_index,
                "num_texts": len(texts_buffer),
            },
            f,
            indent=2,
        )

    print(f"Checkpoint saved: {len(processed_files)} files processed")


def load_checkpoint(output_dir: str):
    """Load checkpoint if exists."""
    checkpoint_path = os.path.join(output_dir, "checkpoints", "checkpoint.pkl")

    if not os.path.exists(checkpoint_path):
        return None

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print(
        f"Loaded checkpoint: {len(checkpoint['processed_files'])} files already processed"
    )
    return checkpoint


def text_iterator_from_parquet(
    dataset_path: str,
    max_files: int = None,
    output_dir: str = None,
    checkpoint_interval: int = 1,
    resume_from_checkpoint: bool = False,
) -> Iterator[str]:
    """Yield text from parquet files one document at a time with checkpointing."""
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))

    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    # Load checkpoint if resuming
    processed_files = []
    texts_buffer = []
    start_index = 0

    if resume_from_checkpoint and output_dir:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            processed_files = checkpoint["processed_files"]
            texts_buffer = checkpoint["texts_buffer"]
            start_index = checkpoint["file_index"]

            # Yield buffered texts first
            print(f"Resuming from checkpoint with {len(texts_buffer)} buffered texts")
            for text in texts_buffer:
                yield text

            # Filter out already processed files
            parquet_files = [f for f in parquet_files if f not in processed_files]
            print(f"Skipping {len(processed_files)} already processed files")

    print(f"Found {len(parquet_files)} parquet files to process")

    texts_buffer = []
    for i, file_path in enumerate(parquet_files, start=start_index):
        print(
            f"Processing file {i+1}/{len(parquet_files) + start_index}: {os.path.basename(file_path)}"
        )

        try:
            table = pq.read_table(file_path)
            texts = table.column("text").to_pylist()

            for text in texts:
                if text:  # Skip empty texts
                    texts_buffer.append(text)
                    yield text

            processed_files.append(file_path)

            # Save checkpoint periodically
            if output_dir and (i + 1) % checkpoint_interval == 0:
                save_checkpoint(output_dir, processed_files, texts_buffer, i + 1)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Save checkpoint on error
            if output_dir:
                save_checkpoint(output_dir, processed_files, texts_buffer, i + 1)
            raise

    # Save final checkpoint
    if output_dir:
        save_checkpoint(
            output_dir, processed_files, texts_buffer, len(parquet_files) + start_index
        )


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
        show_progress=True,
    )

    print(f"Training tokenizer with vocab_size={args.vocab_size}")
    print(f"Reading from: {args.dataset_path}")

    if args.resume_from_checkpoint:
        print("Resume mode enabled - will skip already processed files")

    # Train from iterator
    try:
        tokenizer.train_from_iterator(
            text_iterator_from_parquet(
                args.dataset_path,
                args.max_files,
                output_dir=args.output_dir,
                checkpoint_interval=args.checkpoint_interval,
                resume_from_checkpoint=args.resume_from_checkpoint,
            ),
            trainer=trainer,
        )
    except Exception as e:
        print(f"Error during training: {e}")
        print(
            "Progress has been checkpointed. You can resume with --resume_from_checkpoint"
        )
        raise

    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.output_dir, "tokenizer.json"))

    print(f"Tokenizer saved to {args.output_dir}")

    # Clean up checkpoints after successful training
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        import shutil

        shutil.rmtree(checkpoint_dir)
        print("Checkpoints cleaned up")

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
