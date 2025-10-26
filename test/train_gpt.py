import os
import argparse
import glob
import math
from typing import Iterator, Dict, List
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/Volumes/StorageT4/data/fineweb-edu-parquet-shards/sample-100BT",
        help="Path to directory containing parquet files",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer_output/tokenizer.json",
        help="Path to trained tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gpt_output",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of parquet files to use (None for all)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Context length for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=6,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=384,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=5000,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Evaluation interval",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of iterations for evaluation",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Save checkpoint interval",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory for TensorBoard logs",
    )
    return parser.parse_args()


class GPTConfig:
    """Configuration for GPT model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward through transformer
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens."""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class TextDataset(Dataset):
    """Dataset for loading tokenized text from parquet files."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: Tokenizer,
        block_size: int,
        max_files: int = None,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Load all tokenized sequences
        print("Loading and tokenizing data...")
        self.tokens = []

        parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
        if max_files is not None:
            parquet_files = parquet_files[:max_files]

        for file_path in tqdm(parquet_files, desc="Loading files"):
            table = pq.read_table(file_path)
            texts = table.column("text").to_pylist()

            for text in texts:
                if text:
                    encoding = tokenizer.encode(text)
                    self.tokens.extend(encoding.ids)

        print(f"Loaded {len(self.tokens):,} tokens")

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]

        # Pad if necessary
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [0] * (self.block_size + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


@torch.no_grad()
def estimate_loss(model, train_loader, eval_loader, eval_iters, device):
    """Estimate loss on train and eval sets."""
    out = {}
    model.eval()

    for split, loader in [("train", train_loader), ("eval", eval_loader)]:
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0.0

    model.train()
    return out


def train(args):
    """Train the GPT model."""

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Create config
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    # Create model
    print("Creating model...")
    model = GPT(config)
    model.to(args.device)

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    print(model)

    # Create TensorBoard writer
    log_dir = os.path.join(args.log_dir, f"gpt_{args.output_dir.split('/')[-1]}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to {log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")

    # Log hyperparameters
    writer.add_text("hyperparameters", json.dumps(vars(args), indent=2), 0)
    writer.add_text("model_config", json.dumps(config.__dict__, indent=2), 0)

    # Create datasets
    print("Creating datasets...")
    train_dataset = TextDataset(
        args.dataset_path, tokenizer, args.block_size, args.max_files
    )

    # Split into train and eval (90/10 split)
    train_size = int(0.9 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, eval_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("Starting training...")
    os.makedirs(args.output_dir, exist_ok=True)

    iter_num = 0
    best_eval_loss = float("inf")
    train_loader_iter = iter(train_loader)
    running_loss = 0.0
    log_interval = 10  # Log every 10 iterations

    while iter_num < args.max_iters:
        # Get batch
        try:
            x, y = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            x, y = next(train_loader_iter)

        x, y = x.to(args.device), y.to(args.device)

        # Evaluate periodically
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model, train_loader, eval_loader, args.eval_iters, args.device
            )
            print(
                f"Step {iter_num}: train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}"
            )

            # Log to TensorBoard
            writer.add_scalar("Loss/train", losses["train"], iter_num)
            writer.add_scalar("Loss/eval", losses["eval"], iter_num)
            writer.add_scalar("Loss/best_eval", best_eval_loss, iter_num)

            # Save best model
            if losses["eval"] < best_eval_loss:
                best_eval_loss = losses["eval"]
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_eval_loss": best_eval_loss,
                    "config": config.__dict__,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pt"))
                print(f"Saved best model with eval loss {best_eval_loss:.4f}")

        # Save checkpoint periodically
        if iter_num % args.save_interval == 0 and iter_num > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "config": config.__dict__,
            }
            torch.save(
                checkpoint, os.path.join(args.output_dir, f"checkpoint_{iter_num}.pt")
            )

        # Forward and backward
        _, loss = model(x, y)
        running_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Log gradients
        if iter_num % (args.eval_interval // 2) == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, iter_num)
                    writer.add_histogram(f"parameters/{name}", param, iter_num)

        optimizer.step()

        # Log running loss
        if iter_num % log_interval == 0 and iter_num > 0:
            avg_loss = running_loss / log_interval
            writer.add_scalar("Loss/train_running", avg_loss, iter_num)
            running_loss = 0.0

        iter_num += 1

    # Save final model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "config": config.__dict__,
    }
    torch.save(checkpoint, os.path.join(args.output_dir, "final_model.pt"))

    # Save config as JSON
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)

    print(f"Training complete! Model saved to {args.output_dir}")

    # Generate sample text
    print("\nGenerating sample text...")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=200)
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated: {generated_text}")

    # Log sample generation
    writer.add_text("generated_sample", generated_text, iter_num)

    # Close writer
    writer.close()
    print(f"TensorBoard logs saved to {log_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
