import torch
import os
import argparse
from jamgpt.gpt import GPT, GPTConfig
from jamgpt.common import compute_init

from jamgpt.training import PreTrainer, PreTrainerConfigs
from jamgpt.tokenizer.bpe import BPETokenizer

parser = argparse.ArgumentParser(description="Pretrain GPT model")
parser.add_argument(
    "--tokenizer_model", type=str, required=True, help="Path to tokenizer model"
)
parser.add_argument(
    "--data_dir", type=str, required=True, help="Directory containing training data"
)
parser.add_argument(
    "--num_iterations", type=int, default=-1, help="Number of training iterations"
)
parser.add_argument(
    "--target_param_data_ratio",
    type=int,
    default=20,
    help="Target parameter to data ratio",
)
parser.add_argument(
    "--target_flops", type=float, default=-1.0, help="Target FLOPs for training"
)
parser.add_argument(
    "--total_batch_size", type=int, default=524288, help="Total batch size"
)
parser.add_argument(
    "--num_layers", type=int, default=20, help="Number of transformer layers"
)
parser.add_argument(
    "--aspect_ratio", type=int, default=64, help="Model dimension aspect ratio"
)
parser.add_argument("--sequence_len", type=int, default=2048, help="Sequence length")
parser.add_argument(
    "--unembedding_lr", type=float, default=2e-3, help="Unembedding learning rate"
)
parser.add_argument(
    "--embedding_lr", type=float, default=2e-3, help="Embedding learning rate"
)
parser.add_argument(
    "--matrix_lr", type=float, default=2e-3, help="Matrix learning rate"
)
parser.add_argument(
    "--device_batch_size", type=float, default=32, help="Device batch size"
)
parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay")

args = parser.parse_args()

num_iterations = args.num_iterations
target_param_data_ratio = args.target_param_data_ratio
target_flops = args.target_flops
total_batch_size = args.total_batch_size
num_layers = args.num_layers
model_dim = (
    num_layers * args.aspect_ratio
)  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(
    1, (model_dim + 127) // 128
)  # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads  # 1:1 MQA ratio

print(f"Loading tokenizer from {args.tokenizer_model}...")
tokenizer = BPETokenizer.from_directory(args.tokenizer_model)


model_config = GPTConfig(
    sequence_len=args.sequence_len,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)

pretraining_config = PreTrainerConfigs(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
    device_batch_size=args.device_batch_size,
    data_dir=args.data_dir,
)

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
token_bytes_path = os.path.join(args.tokenizer_model, "token_bytes.pt")
with open(token_bytes_path, "rb") as f:
    token_bytes = torch.load(f, map_location=device)

with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device="cuda")
model.init_weights()
orig_model = model  # original, uncompiled model, for saving raw model state_dict
model = torch.compile(model, dynamic=False)  # TODO: dynamic True/False think through
num_params = sum(p.numel() for p in model.parameters())

assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    # use the specified number of iterations
    pass
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (model.estimate_flops() * total_batch_size))
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations


trainer = PreTrainer(
    config=pretraining_config,
    model=model,
    num_iterations=num_iterations,
    tokenizer=tokenizer,
)
trainer.train()
