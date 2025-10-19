import os
import time
import argparse
import torch

from jamgpt.tokenizer.bpe import BPETokenizer
from jamgpt.common import get_base_dir, parquets_iter_batched


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--max_chars",
        type=int,
        default=10_000_000_000,
        help="Maximum characters to train on (default: 10B)",
    )
    parser.add_argument(
        "--doc_cap",
        type=int,
        default=10_000,
        help="Maximum characters per document (default: 10,000)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=65536,
        help="Vocabulary size (default: 65536 = 2^16)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT",
        help="Directory containing parquet data files",
    )
    return parser.parse_args()


def text_iterator(max_chars, doc_cap, data_dir: str):
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to doc_cap characters
    3) Break when we've seen max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(data_dir=data_dir, split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > doc_cap:
                doc_text = doc_text[:doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > max_chars:
                return


def save_token_bytes(tokenizer, tokenizer_dir):
    """Save token byte lengths for each token in the vocabulary."""
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []

    for token_id in range(vocab_size):
        token_str = token_strings[token_id]
        if token_str in special_set:
            token_bytes.append(0)  # special characters are not counted
        else:
            id_bytes = len(token_str.encode("utf-8"))
            token_bytes.append(id_bytes)

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")


def main():
    args = parse_args()

    print(f"Training tokenizer with the following parameters:")
    print(f"  Max characters: {args.max_chars:,}")
    print(f"  Document cap: {args.doc_cap:,}")
    print(f"  Vocab size: {args.vocab_size:,}")

    # Create text iterator
    text_iter = text_iterator(args.max_chars, args.doc_cap, args.data_dir)

    # Train tokenizer
    print("\nStarting tokenizer training...")
    start_time = time.time()
    tokenizer = BPETokenizer.train_from_iterator(text_iter, args.vocab_size)
    end_time = time.time()
    print(f"Tokenizer training completed in {end_time - start_time:.2f} seconds.")

    # Save tokenizer
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_to_directory(tokenizer_dir)
    print(f"Tokenizer saved to {tokenizer_dir}")

    save_token_bytes(tokenizer, tokenizer_dir)


if __name__ == "__main__":
    main()
