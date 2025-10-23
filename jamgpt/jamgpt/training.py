from dataclasses import dataclass


@dataclass
class PreTrainerConfigs:
    unembedding_lr: float
    embedding_lr: float
    matrix_lr: float
    weight_decay: float
    device_batch_size: float
    data_dir: str
    tokenizer_dir: str
