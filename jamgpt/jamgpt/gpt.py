from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from functools import partial

from torch import nn
from jamgpt.common import get_dist_info

from jamgpt.optimizers import Muon, DistMuon, DistAdamW, Muon


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    sequence_len: int = 1024
    n_layers: int = 24
    n_heads: int = 6  # query heads
    n_kv_heads: int = 6  # key/value heads
    n_embd: int = 768
    head_dim: int = 768 // 6


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.c_q = nn.Linear(
            config.n_embd, config.n_heads * config.head_dim, bias=False
        )
        self.c_k = nn.Linear(
            config.n_embd, config.n_kv_heads * config.head_dim, bias=False
        )
        self.c_v = nn.Linear(
            config.n_embd, config.n_kv_heads * config.head_dim, bias=False
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def norm(x, eps=1e-6):
        """Layer normalization over the last dimension."""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + eps)

    def apply_rotary_emb(self, x, cos, sin):
        """Apply rotary embeddings to tensor x."""
        d = x.shape[3] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=3).to(x.dtype)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, H, self.config.n_heads)
        k = self.c_k(x).view(B, T, self.config.n_kv_heads, self.config.head_dim)
        v = self.c_v(x).view(B, T, self.config.n_kv_heads, self.config.head_dim)

        # Apply rotary embeddings
        cos, sin = cos_sin
        q = self.apply_rotary_emb(q, cos, sin)
        k = self.apply_rotary_emb(k, cos, sin)
        q, k = self.norm(q), self.norm(k)

        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, H, T, D)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)

        # MQA
        nrep = self.config.n_heads // self.config.n_kv_heads
        k, v = self.repeat_kv(k, nrep), self.repeat_kv(v, nrep)

        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        elif Tq == 1:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            if prefix_len > 0:
                attn_mask[:, :prefix_len] = True
            attn_mask = torch.tril(torch.ones((Tq, Tk), device=q.device)).to(torch.bool)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.fc2 = nn.Linear(config.n_embd * 4, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(self.ln1(x), cos_sin, kv_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layers)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # fake init
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.transformer.wte.to(dtype=torch.float16)

    def initialize_weights(self):
        self.apply(self._initialize_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim
        )

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.range(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(position, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def setup_optimizers(
        self,
        unembedding_lr: float,
        embedding_lr: float,
        matrix_lr: float,
        weight_decay: float,
    ):
        ddp, rank, local_rank, world_size, device = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        if rank == 0:
            print(
                f"Setting matrix learning rate scale to {dmodel_lr_scale:.4f} based on model dimension {self.config.n_embd}"
            )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(weight_decay=weight_decay, betas=(0.8, 0.95), eps=1e-10)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, **adamw_kwargs)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else partial(Muon, **muon_kwargs)
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["initial_lr"] = param_group["lr"]
        return optimizers

    def norm(x, eps=1e-6):
        """Layer normalization over the last dimension."""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + eps)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()
        assert (
            T <= self.config.sequence_len
        ), "Cannot forward, model sequence length exceeded."
        assert (
            idx.device == self.cos.device
        ), "Input device does not match model device."
        assert (
            self.cos.dtype == torch.bfloat16
        ), "Rotary embeddings must be in bfloat16."

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )
        x = self.transformer.wte(idx)  # token embeddings
        x = self.norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = self.norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)

        if targets is None:
            return logits
        else:
            logits = logits.float()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return logits, loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Generate tokens autoregressively."""
        device = self.transformer.wte.weight.device
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        rng = (
            None
            if temperature <= 0
            else torch.Generator(device=device).manual_seed(seed)
        )
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_ids], dim=1)
            token = next_ids.item()
            yield token
