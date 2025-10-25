"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import argparse
from contextlib import nullcontext
from tqdm import tqdm

# import wandb
import torch

from jamgpt.gpt import GPT, GPTConfig
from jamgpt.dataloader import tokenizing_distributed_data_loader
from jamgpt.common import (
    compute_init,
    compute_cleanup,
    print0,
    # DummyWandb,
    autodetect_device_type,
)
from jamgpt.tokenizer import RustBPETokenizer, get_token_bytes
from jamgpt.checkpoint_manager import save_checkpoint
from jamgpt.loss_eval import evaluate_bpb
from jamgpt.engine import Engine
from scripts.base_eval import evaluate_model


parser = argparse.ArgumentParser(description="Train GPT model")
parser.add_argument(
    "--tokenizer_dir",
    type=str,
    required=True,
    help="Path to tokenizer model directory",
)
# Runtime
parser.add_argument(
    "--device_type",
    type=str,
    default="",
    help="Device type: cuda|cpu|mps (empty = autodetect)",
)
# Model architecture
parser.add_argument(
    "--depth", type=int, default=20, help="Depth of the Transformer model"
)
parser.add_argument("--max_seq_len", type=int, default=2048, help="Max context length")
# Training horizon
parser.add_argument(
    "--num_iterations",
    type=int,
    default=-1,
    help="Explicit number of optimization steps (-1 = disable)",
)
parser.add_argument(
    "--target_flops",
    type=float,
    default=-1.0,
    help="Calculate num_iterations to reach target FLOPs (-1 = disable)",
)
parser.add_argument(
    "--target_param_data_ratio",
    type=int,
    default=20,
    help="Calculate num_iterations for fixed data:param ratio (-1 = disable)",
)
# Optimization
parser.add_argument(
    "--device_batch_size", type=int, default=32, help="Per-device batch size"
)
parser.add_argument(
    "--total_batch_size",
    type=int,
    default=524288,
    help="Total desired batch size in tokens",
)
parser.add_argument(
    "--embedding_lr",
    type=float,
    default=0.2,
    help="Learning rate for embedding parameters (Adam)",
)
parser.add_argument(
    "--unembedding_lr",
    type=float,
    default=0.004,
    help="Learning rate for unembedding parameters (Adam)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Weight decay for embedding/unembedding parameters (Adam)",
)
parser.add_argument(
    "--matrix_lr",
    type=float,
    default=0.02,
    help="Learning rate for matrix parameters (Muon)",
)
parser.add_argument(
    "--grad_clip",
    type=float,
    default=1.0,
    help="Gradient clipping value (0.0 = disabled)",
)
# Evaluation
parser.add_argument(
    "--eval_every", type=int, default=250, help="Evaluate val bpb every N steps"
)
parser.add_argument(
    "--eval_tokens",
    type=int,
    default=20 * 524288,
    help="Number of tokens for val loss evaluation",
)
parser.add_argument(
    "--core_metric_every",
    type=int,
    default=2000,
    help="Evaluate core metric every N steps (-1 = disable)",
)
parser.add_argument(
    "--core_metric_max_per_task",
    type=int,
    default=500,
    help="Examples per task for core metric",
)
parser.add_argument(
    "--sample_every", type=int, default=2000, help="Sample from model every N steps"
)
# Output
parser.add_argument(
    "--checkpoints_path",
    type=str,
    required=True,
    help="Path to the directory where checkpoints are saved",
)
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to the dataset directory"
)

args = parser.parse_args()

# Print all arguments
print("Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print()

# Map args to original variable names for compatibility
device_type = args.device_type
depth = args.depth
max_seq_len = args.max_seq_len
num_iterations = args.num_iterations
target_flops = args.target_flops
target_param_data_ratio = args.target_param_data_ratio
device_batch_size = args.device_batch_size
total_batch_size = args.total_batch_size
embedding_lr = args.embedding_lr
unembedding_lr = args.unembedding_lr
weight_decay = args.weight_decay
matrix_lr = args.matrix_lr
grad_clip = args.grad_clip
eval_every = args.eval_every
eval_tokens = args.eval_tokens
core_metric_every = args.core_metric_every
core_metric_max_per_task = args.core_metric_max_per_task
sample_every = args.sample_every

# Create user_config from args for logging
user_config = vars(args)
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
# use_dummy_wandb = run == "dummy" or not master_process
# wandb_run = (
#     DummyWandb()
#     if use_dummy_wandb
#     else wandb.init(project="nanochat", name=run, config=user_config)
# )

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = RustBPETokenizer.from_directory(args.tokenizer_dir)
token_bytes = get_token_bytes(tokenizer_dir=args.tokenizer_dir, device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = (
    depth * 64
)  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(
    1, (model_dim + 127) // 128
)  # head dim 128 (the division here is ceil div)
num_kv_heads = (
    num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = (
    device_batch_size * max_seq_len
)  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = (
    tokens_per_fwdbwd * ddp_world_size
)  # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
)
# -----------------------------------------------------------------------------
# Initialize the Model
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
orig_model = model  # original, uncompiled model, for saving raw model state_dict
model = torch.compile(model, dynamic=False)  # TODO: dynamic True/False think through
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(
        f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
    )
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(
    f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}"
)  # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

train_loader = tokenizing_distributed_data_loader(
    args.tokenizer_dir,
    device_batch_size,
    max_seq_len,
    split="train",
    device=device,
    dataset_path=args.dataset_path,
)

build_val_loader = lambda: tokenizing_distributed_data_loader(
    args.tokenizer_dir,
    device_batch_size,
    max_seq_len,
    split="val",
    device=device,
    dataset_path=args.dataset_path,
)

x, y = next(train_loader)  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler

warmup_ratio = 0.0  # ratio of iterations for LR warmup
warmdown_ratio = 0.2  # ratio of iterations for LR warmdown
final_lr_frac = 0.0  # final LR is this fraction of the initial LR


def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def muon_momentum_scheduler(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


min_val_bpb = float("inf")
smooth_train_loss = 0  # EMA of training loss
ema_beta = 0.9  # EMA decay factor
total_training_time = 0  # total wall-clock time of training
# note that we run +1 steps only so that we can eval and save at the end
for step in tqdm(
    range(num_iterations + 1), desc="Training", disable=not master_process
):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        # wandb_run.log(
        #     {
        #         "step": step,
        #         "total_training_flops": flops_so_far,
        #         "total_training_time": total_training_time,
        #         "val/bpb": val_bpb,
        #     }
        # )
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (
        last_step or (step > 0 and step % core_metric_every == 0)
    ):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(
                orig_model, tokenizer, device, max_per_task=core_metric_max_per_task
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        # wandb_run.log(
        #     {
        #         "step": step,
        #         "total_training_flops": flops_so_far,
        #         "core_metric": results["core_metric"],
        #         "centered_results": results["centered_results"],
        #     }
        # )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step:
        save_checkpoint(
            args.checkpoints_path,
            step,
            orig_model.state_dict(),
            [
                opt.state_dict() for opt in optimizers
            ],  # TODO: make sure saving across ranks is done correctly
            {
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config,  # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            },
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # for logging
        loss = (
            loss / grad_accum_steps
        )  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
    # gradient clipping (TODO possibly expertiment with)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = muon_momentum_scheduler(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    )  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (
        1 - ema_beta ** (step + 1)
    )  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = (
        989e12 * ddp_world_size
    )  # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # in %
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m"
    )
    # if step % 100 == 0:
    #     wandb_run.log(
    #         {
    #             "step": step,
    #             "total_training_flops": flops_so_far,
    #             "total_training_time": total_training_time,
    #             "train/loss": debiased_smooth_loss,
    #             "train/lrm": lrm,
    #             "train/dt": dt,
    #             "train/tok_per_sec": tok_per_sec,
    #             "train/mfu": mfu,
    #         }
    #     )

print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")


# wandb_run.finish()  # wandb run finish
compute_cleanup()
