#!/usr/bin/env python
"""
Train a GPT-style decoder-only LM (GPT-2-ish) from scratch on FineWeb-Edu.
- Trains a byte-level BPE tokenizer from a stream of the dataset
- Builds a GPT-2 style config and model sized by flags
- Streams "HuggingFaceFW/fineweb-edu" and packs tokens into fixed-length blocks
- Uses Hugging Face Transformers Trainer for CLM

Tested with:
  transformers >= 4.43
  datasets >= 2.18
  tokenizers >= 0.15
  accelerate >= 0.30
  torch >= 2.2

Example:
  python train_gpt_from_fineweb_edu.py \
    --model_size small \
    --tokenizer_size 50257 \
    --seq_len 1024 \
    --tokenizer_train_chars 100_000_000 \
    --train_steps 20_000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --eval_steps 500 \
    --save_steps 2000 \
    --lr 3e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --output_dir runs/gpt2-small-fineweb-edu

Notes:
- "GPT-3 quality" requires scale; this script scales up (n_layer/n_head/n_embd) but budget is on you.
- We stream the dataset to avoid local storage. For fast tokenizer training, a char budget is used.
- For real runs, set --tokenizer_train_chars into the billions and consider multi-node training.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Dict, Any, Optional

import datasets as hfds
from datasets import IterableDataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Sequence, NFD, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import default_data_collator


# ------------------------------
# Tokenizer training
# ------------------------------


def iter_text_stream(dataset: IterableDataset, char_budget: int) -> Iterator[str]:
    """Yield text strings up to a character budget from a streaming dataset."""
    seen = 0
    for ex in dataset:
        txt = ex.get("text")
        if not txt:
            continue
        yield txt
        seen += len(txt)
        if seen >= char_budget:
            break


def build_or_load_tokenizer(
    out_dir: str,
    vocab_size: int = 50257,
    char_budget: int = 50_000_000,
    min_frequency: int = 2,
    lowercase: bool = False,
    special_tokens: Optional[List[str]] = None,
) -> PreTrainedTokenizerFast:
    os.makedirs(out_dir, exist_ok=True)
    tok_path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"[tokenizer] Loading existing tokenizer from {tok_path}")
        tok = Tokenizer.from_file(tok_path)
    else:
        print("[tokenizer] Training byte-level BPE from iterator…")
        # Byte-level BPE model, GPT-2 style
        tok = Tokenizer(BPE(unk_token="<|unk|>"))
        normalizers = []
        # GPT-2 didn't lowercase; allow toggling for experimentation
        if lowercase:
            normalizers.append(NFD())
            normalizers.append(Lowercase())
        tok.normalizer = Sequence(normalizers) if normalizers else None
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()

        if special_tokens is None:
            special_tokens = [
                "<|bos|>",
                "<|eos|>",
                "<|pad|>",
                "<|unk|>",
                "<|sep|>",
                "<|mask|>",
            ]

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        # Stream a subset of FineWeb-Edu to train the tokenizer
        stream = hfds.load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        tok.train_from_iterator(iter_text_stream(stream, char_budget), trainer=trainer)
        tok.save(tok_path)
        print(f"[tokenizer] Saved tokenizer to {tok_path}")

    # Wrap in HF fast tokenizer API
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    # Ensure special token mapping exists
    fast.add_special_tokens(
        {
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
        }
    )
    return fast


# ------------------------------
# Iterable dataset that packs tokens
# ------------------------------


@dataclass
class PackingConfig:
    seq_len: int = 1024


class PackedCLMIterable(hfds.IterableDataset):
    """
    Streams raw text -> tokenizes -> packs into fixed-length blocks for causal LM.

    Yields dicts with input_ids, attention_mask, labels (labels == input_ids).
    """

    def __init__(
        self,
        text_stream: IterableDataset,
        tokenizer: PreTrainedTokenizerFast,
        pack_cfg: PackingConfig,
    ):
        super().__init__()
        self.text_stream = text_stream
        self.tokenizer = tokenizer
        self.seq_len = pack_cfg.seq_len

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        buffer: List[int] = []
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        assert bos_id is not None and eos_id is not None

        for ex in self.text_stream:
            txt = ex.get("text")
            if not txt:
                continue
            # Add BOS/EOS around documents to reduce cross-doc bleed
            ids = self.tokenizer.encode(txt, add_special_tokens=False)
            buffer.extend([bos_id] + ids + [eos_id])

            # Emit fixed-size chunks
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                del buffer[: self.seq_len]
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * self.seq_len,
                    "labels": chunk.copy(),
                }

        # Drop the tail by default; could optionally emit padded tail


# ------------------------------
# Model factory
# ------------------------------


def model_config_for_size(size: str, vocab_size: int, seq_len: int) -> GPT2Config:
    sizes = {
        # (n_layer, n_head, n_embd)
        "tiny": (4, 4, 256),
        "mini": (6, 6, 384),
        "small": (12, 12, 768),  # GPT-2 small-ish
        "medium": (24, 16, 1024),
        "large": (36, 20, 1280),
        "xl": (48, 25, 1600),  # approaching GPT-3 Ada scale (still far from 175B)
    }
    if size not in sizes:
        raise ValueError(f"Unknown model_size '{size}'. Choices: {list(sizes.keys())}")
    n_layer, n_head, n_embd = sizes[size]
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,  # will be overwritten after tokenizer resize
        eos_token_id=0,
    )
    return cfg


# ------------------------------
# Train
# ------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="tiny|mini|small|medium|large|xl",
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=None,
        help="If provided, load tokenizer from this dir; else train new in output_dir/tokenizer",
    )
    parser.add_argument("--tokenizer_size", type=int, default=50257)
    parser.add_argument("--tokenizer_min_frequency", type=int, default=2)
    parser.add_argument("--tokenizer_train_chars", type=int, default=50_000_000)
    parser.add_argument("--tokenizer_lowercase", action="store_true")

    # Trainer / optim
    parser.add_argument("--train_steps", type=int, default=100_000)
    parser.add_argument("--eval_steps", type=int, default=1_000)
    parser.add_argument("--save_steps", type=int, default=10_000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Tokenizer
    tok_dir = args.tokenizer_dir or os.path.join(args.output_dir, "tokenizer")
    tokenizer = build_or_load_tokenizer(
        out_dir=tok_dir,
        vocab_size=args.tokenizer_size,
        char_budget=args.tokenizer_train_chars,
        min_frequency=args.tokenizer_min_frequency,
        lowercase=args.tokenizer_lowercase,
    )

    # 2) Model
    cfg = model_config_for_size(args.model_size, tokenizer.vocab_size, args.seq_len)
    model = GPT2LMHeadModel(cfg)
    # Ensure model knows correct special token ids and resize embeddings
    num_added = model.resize_token_embeddings(len(tokenizer))
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    # GPT-2 has no pad; map to eos to avoid -100 spam
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 3) Data: streaming FineWeb-Edu train/eval
    train_stream = hfds.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )
    # For eval, take a small slice (deterministic by seed); we rely on streaming order stability
    eval_stream = (
        hfds.load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )
        .skip(1_000_000)
        .take(10_000)
    )  # avoid overlap; tune as needed

    pack_cfg = PackingConfig(seq_len=args.seq_len)
    train_ds = PackedCLMIterable(train_stream, tokenizer, pack_cfg)
    eval_ds = PackedCLMIterable(eval_stream, tokenizer, pack_cfg)

    # 4) Trainer
    total_train_batch = args.per_device_train_batch_size
    # Trainer with iterable datasets needs max_steps
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.train_steps,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["tensorboard"],
        dataloader_drop_last=True,  # packed sequences are fixed-length
        remove_unused_columns=False,  # we already emit exact fields
        save_total_limit=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,  # labels already provided
    )

    print("[train] Starting training…")
    trainer.train()
    print("[train] Training finished. Saving…")

    # 5) Save artifacts
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        f.write(model.config.to_json_string())

    print("[done] Artifacts saved to", args.output_dir)


if __name__ == "__main__":
    main()
