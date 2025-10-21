import torch
import os
from jamgpt.gpt import GPT, GPTConfig
from jamgpt.common import compute_init, num_flops_per_token

from jamgpt.training import PreTrainer, PreTrainerConfigs
from jamgpt.tokenizer.bpe import get_tokenizer, get_token_bytes

num_iterations = -1
target_param_data_ratio = 20
target_flops = -1.0
total_batch_size = 524288
num_layers = 20
model_dim = (
    num_layers * 64
)  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(
    1, (model_dim + 127) // 128
)  # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads  # 1:1 MQA ratio

tokenizer = get_tokenizer()

model_config = GPTConfig(
    sequence_len=2048,
    vocab_size=tokenizer.get_vocab_size(),
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)

pretraining_config = PreTrainerConfigs(
    unembedding_lr=2e-3,
    embedding_lr=2e-3,
    matrix_lr=2e-3,
    weight_decay=1e-1,
)

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
token_bytes = get_token_bytes(device=device)

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
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations


trainer = PreTrainer(
    config=pretraining_config, model=model, num_iterations=num_iterations
)
trainer.train()
