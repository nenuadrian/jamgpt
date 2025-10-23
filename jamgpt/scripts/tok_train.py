import os
import time
import argparse
import torch
from jamgpt.tokenizer import RustBPETokenizer
from jamgpt.dataloader import parquets_iter_batched


def main():
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
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory containing parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained tokenizer model",
    )
    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    def text_iterator():
        """
        1) Flatten the batches into a single iterator
        2) Crop every document to args.doc_cap characters
        3) Break when we've seen args.max_chars characters
        """
        nchars = 0
        for batch in parquets_iter_batched(
            split="train", dataset_path=args.dataset_path
        ):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[: args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return

    text_iter = text_iterator()

    # -----------------------------------------------------------------------------
    # Train the tokenizer
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    tokenizer.save(args.output_dir)

    # -----------------------------------------------------------------------------
    # Quick inline sanity check
    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

    # -----------------------------------------------------------------------------
    # One more thing: we wish to cache a mapping from token id to number of bytes of that token
    # for efficient evaluation of bits per byte. Unlike the typical mean loss, this
    # allows us to report a loss that is invariant to the vocab size of the tokenizer.
    # The bits per byte on the validation set is then one of the primary metrics we care about.
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = token_strings[
            token_id
        ]  # the Python string representation of this token
        if token_str in special_set:
            token_bytes.append(0)  # special characters are not counted
        else:
            id_bytes = len(
                token_str.encode("utf-8")
            )  # number of bytes that make up this token
            token_bytes.append(id_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    token_bytes_path = os.path.join(args.output_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")


if __name__ == "__main__":
    main()
