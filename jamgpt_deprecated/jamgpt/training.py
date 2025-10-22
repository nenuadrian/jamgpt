import math
from dataclasses import dataclass
from jamgpt.gpt import GPT
from jamgpt.dataloader import tokenizing_distributed_data_loader

WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.2
FINAL_LR_FRAC = 0.0


@dataclass
class PreTrainerConfigs:
    unembedding_lr: float
    embedding_lr: float
    matrix_lr: float
    weight_decay: float
    device_batch_size: float
    data_dir: str


class PreTrainer:
    def __init__(
        self, config: PreTrainerConfigs, model: GPT, tokenizer, num_iterations: int
    ):
        self.model = model
        self.num_iterations = num_iterations

        self.adamw_optimizer, self.muon_optimizer = model.setup_optimizers(
            unembedding_lr=config.unembedding_lr,
            embedding_lr=config.embedding_lr,
            matrix_lr=config.matrix_lr,
            weight_decay=config.weight_decay,
        )

        self.train_loader = tokenizing_distributed_data_loader(
            B=config.device_batch_size,
            T=config.sequence_len,
            split="train",
            data_dir=config.data_dir,
            tokenizer=tokenizer,
        )
        self.build_val_loader = lambda: tokenizing_distributed_data_loader(
            B=config.device_batch_size,
            T=config.sequence_len,
            split="val",
            data_dir=config.data_dir,
            tokenizer=tokenizer,
        )

    def train(self):
        x, y = next(self.train_loader)

    def get_learning_rate_multiplier(self, it):
        """Get learning rate multiplier for a given iteration."""
        WARMUP_ITERS = int(self.num_iterations * WARMUP_RATIO)
        WARMDOWN_ITERS = int(self.num_iterations * WARMDOWN_RATIO)
        if it < WARMUP_ITERS:
            return (it + 1) / WARMUP_ITERS
        elif it > self.num_iterations - WARMDOWN_ITERS:
            down_it = it - (self.num_iterations - WARMDOWN_ITERS)
            down_ratio = down_it / WARMDOWN_ITERS
            cosine_decay = 0.5 * (1 + math.cos(math.pi * down_ratio))
            return FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * cosine_decay
        else:
            return 1.0

    def get_muon_momentum(self, it):
        frac = min(it / 300, 1)
        momentum = (1 - frac) * 0.95 + frac * 0.95
        return momentum
