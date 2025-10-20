"""
Comparing the training of:

HuggingFace tokenizers training implementation
Our own custom RustBPE training implementation

All of these should calculate the same merges and produce
the same vocabulary and tokenizations.

Finally, for inference we will use tiktoken for efficiency.
So we want to make sure we can export our rustbpe tokenizer
into tiktoken and use it for inference with identical results.
"""

import time
import tiktoken
import pytest
import requests
from jamgpt import rustbpe
from jamgpt import BPETokenizer

import tempfile


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# -----------------------------------------------------------------------------
# HuggingFace tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(
            BPE(
                byte_fallback=True,  # needed!
                unk_token=None,
                fuse_unk=False,
            )
        )
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        gpt4_split_regex = Regex(
            GPT4_SPLIT_PATTERN
        )  # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=gpt4_split_regex, behavior="isolated", invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,  # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],  # no special tokens
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def encode_ordinary(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids


# -----------------------------------------------------------------------------
# Test all of the above


@pytest.fixture(scope="module")
def enwik8_path():
    """Fixture to download and cache enwik8 dataset."""
    import os
    import zipfile
    from jamgpt.common import get_base_dir

    base_dir = get_base_dir()
    # download and unzip enwik8 to .cache directory
    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = os.path.join(base_dir, "enwik8")
    enwik8_local_path_zip = os.path.join(base_dir, "enwik8.zip")
    if not os.path.exists(enwik8_local_path):
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")

        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        os.remove(enwik8_local_path_zip)
        print(f"Removed {enwik8_local_path_zip}")
    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path


@pytest.fixture(scope="module")
def enwik8_small(enwik8_path):
    """Fixture providing 100KB of enwik8 for quick tests."""
    with open(enwik8_path, "r") as f:
        return f.read(100_000)


@pytest.fixture(scope="module")
def enwik8_large(enwik8_path):
    """Fixture providing 10MB of enwik8 for performance tests."""
    with open(enwik8_path, "r") as f:
        return f.read(10**7)


def time_function(func, *args, **kwargs):
    """Time a function call and return the result and elapsed time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed


def test_correctness(enwik8_small):
    """Test that all tokenizer implementations produce the same results."""
    text = enwik8_small
    encode_text = text
    vocab_size = 256 + 20  # 20 merges

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(
        HuggingFaceTokenizer.train_from_iterator, [text], vocab_size
    )
    hf_ids, hf_encode_time = time_function(hf_tokenizer.encode_ordinary, encode_text)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    print(f"HuggingFace encode time: {hf_encode_time:.4f}s")
    print(hf_ids[:20])

    # HuggingFace has a different byte order, so we need custom matching
    def custom_match(ids1, ids2):
        perm = {}
        for x, y in zip(ids1, ids2):
            if x < 256:
                if x in perm:
                    if perm[x] != y:
                        return False
                perm[x] = y
            if x >= 256 and x != y:
                return False
        return True

    # Finally use our own Rust implementation
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(
        rustbpe_tokenizer.train_from_iterator, [text], vocab_size
    )
    rustbpe_ids, rustbpe_encode_time = time_function(
        rustbpe_tokenizer.encode, encode_text
    )
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    print(f"RustBPE encode time: {rustbpe_encode_time:.4f}s")
    print(rustbpe_ids[:20])

    assert custom_match(hf_ids, rustbpe_ids), "HuggingFace should match RustBPE"
    print("✅ RustBPE == Fast")

    # Now export rustbpe to tiktoken for more efficient inference
    print("\nTesting tiktoken export...")
    pattern = rustbpe_tokenizer.get_pattern()
    mergeable_ranks_list = rustbpe_tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    tiktoken_ids, tiktoken_encode_time = time_function(enc.encode, encode_text)
    print(f"Tiktoken encode time: {tiktoken_encode_time:.4f}s")
    print(tiktoken_ids[:20])

    assert tiktoken_ids == rustbpe_ids, "Tiktoken should match RustBPE"
    print("✅ Tiktoken == RustBPE")


@pytest.mark.slow
def test_training_performance(enwik8_large):
    """Use a bigger dataset and compare the training speed of the optimized tokenizers (Python, Rust, HuggingFace)."""
    text = enwik8_large
    vocab_size = 2048
    print(f"\nText length: {len(text)}")

    # Commenting out because it's just way too slow to matter
    # Train optimized python version
    # print("Training optimized python version...")
    # optimized_python_tokenizer = FastRegexTokenizer()
    # _, optimized_python_train_time = time_function(optimized_python_tokenizer.train, text, vocab_size)
    # print(f"Optimized python train time: {optimized_python_train_time:.4f}s")

    # Train rustbpe
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(
        rustbpe_tokenizer.train_from_iterator, [text], vocab_size
    )
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    assert rustbpe_train_time > 0, "Training should take some time"

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(
        HuggingFaceTokenizer.train_from_iterator, [text], vocab_size
    )
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    assert hf_train_time > 0, "Training should take some time"

    # Print comparison
    print(f"\n📊 Performance comparison:")
    print(f"   RustBPE: {rustbpe_train_time:.4f}s")
    print(f"   HuggingFace: {hf_train_time:.4f}s")
    print(f"   Speedup: {hf_train_time/rustbpe_train_time:.2f}x")


def test_interface(enwik8_small):
    """Test the BPETokenizer interface for training, encoding, decoding, and serialization."""

    # Simple train test
    vocab_size = 300
    tok = BPETokenizer.train_from_iterator([enwik8_small], vocab_size)
    assert (
        tok.get_vocab_size() == vocab_size
    ), f"Expected vocab size {vocab_size}, got {tok.get_vocab_size()}"
    print(f"✅ Trained tokenizer with vocab size {vocab_size}")

    # Encode/decode text
    encode_text = "Hello world! How are you? 🙃"
    ids = tok.encode([encode_text])
    print(f"\nInput text: {encode_text}")
    print(f"IDs: {ids}")
    decoded = tok.decode(ids[0])
    print(f"Decoded: {decoded}")
    assert (
        decoded == encode_text
    ), f"Decoded text doesn't match: {decoded} != {encode_text}"
    print("✅ Encode/decode test passed")

    # Encode batch test
    ids_new = tok.encode([encode_text, encode_text])
    assert all(
        x == ids[0] for x in ids_new
    ), "Batch encoding should produce identical results"
    print("✅ Encode batch OK")

    # append/prepend functionality
    ids_special = tok.encode([encode_text], prepend="<|bos|>", append="<|bos|>")
    bos_token_id = tok.encode_special("<|bos|>")
    assert ids_special[0] == [bos_token_id] + ids[0] + [
        bos_token_id
    ], "Special tokens not correctly added"
    print("✅ append/prepend OK")

    # Save/load test through a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tok.save_to_directory(tmp_dir)
        tok_reloaded = BPETokenizer.from_directory(tmp_dir)
        ids_reloaded = tok_reloaded.encode([encode_text])
        assert ids_reloaded == ids, "Reloaded tokenizer should produce same results"
        print("✅ Save/load through temporary directory OK")
