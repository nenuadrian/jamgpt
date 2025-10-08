from __future__ import annotations
from typing import List, Tuple
import torch

from tokenizer_bpe import BPETokenizer

from formatters import Example, format_example


class PairCollator:
    """Tokenize preference pairs into (pos, neg) input ids.
    We format as the SFT template with the 'chosen' or 'rejected' text as the Response.
    """

    def __init__(
        self,
        block_size: int = 256,
        bpe_dir: str | None = None,
        vocab_size: int | None = None,
    ):
        self.block_size = block_size
        self.tok = None
        try:
            self.tok = BPETokenizer(vocab_size=vocab_size or 8000)
            if bpe_dir:
                self.tok.load(bpe_dir)
        except Exception:
            self.tok = None
        if self.tok is None:
            raise RuntimeError("No tokenizer available.")

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, "vocab_size", 256)

    def _encode(self, text: str) -> List[int]:
        if hasattr(self.tok, "encode"):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        return list(text.encode("utf-8"))

    def collate(self, batch: List[Tuple[str, str, str]]):
        # batch of (prompt, chosen, rejected)
        pos_ids, neg_ids = [], []
        for prompt, chosen, rejected in batch:
            pos_text = format_example(Example(prompt, chosen))
            neg_text = format_example(Example(prompt, rejected))
            pos_ids.append(self._encode(pos_text)[: self.block_size])
            neg_ids.append(self._encode(neg_text)[: self.block_size])

        def pad_to(x, pad=2):
            return (
                x + [pad] * (self.block_size - len(x))
                if len(x) < self.block_size
                else x[: self.block_size]
            )

        pos = torch.tensor([pad_to(x) for x in pos_ids], dtype=torch.long)
        neg = torch.tensor([pad_to(x) for x in neg_ids], dtype=torch.long)
        return pos, neg
