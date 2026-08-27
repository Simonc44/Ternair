"""Light-weight data loading for pre-training ternary models.

Supports HuggingFace ``datasets`` with optional streaming, as well as
the self-contained toy corpus for smoke tests.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ternair.training.config import TrainingConfig

_LOGGER = logging.getLogger(__name__)

DEFAULT_CORPUS = """the quick brown fox jumps over the lazy dog
the rain in spain falls mainly on the plain
hello world this is a tiny prototype
ternair is a bitnet b1 58 style transformer
press green button and the magic happens
gpt is not a database gpt is a simulator
"""


def toy_corpus() -> str:
    return DEFAULT_CORPUS


class CharTokenizer:
    """Minimal character-level tokenizer (for smoke tests)."""

    def __init__(self, text: str | None = None) -> None:
        text = text if text is not None else DEFAULT_CORPUS
        chars = sorted(set(text))
        self.control = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        next_id = max(self.control.values()) + 1
        self.token_to_id = dict(self.control)
        for ch in chars:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = next_id
                next_id += 1
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = max(self.token_to_id.values()) + 1

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(ch, self.token_to_id["<unk>"]) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            t = self.id_to_token.get(int(i), "<unk>")
            if t in ("<pad>", "<bos>", "<eos>", "<unk>"):
                continue
            out.append(t)
        return "".join(out)

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]


class ToyDataset(torch.utils.data.Dataset):
    """Repeated toy-corpus sequences for multi-step smoke training.

    Wraps :func:`tokenise_corpus` so a real training loop can run
    several optimizer steps without requiring the HuggingFace ``datasets``
    dependency (useful for CPU smoke tests and CI).
    """

    def __init__(self, text: str | None = None, n_sequences: int = 64) -> None:
        ids, _ = tokenise_corpus(text=text, max_len=256)
        self.seq = ids.squeeze(0)
        self.n_sequences = n_sequences

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, index: int) -> dict:
        return {"input_ids": self.seq, "labels": self.seq}


def build_toy_dataloader(
    text: str | None = None,
    n_sequences: int = 64,
    batch_size: int = 8,
    repeat: bool = True,
) -> DataLoader:
    """DataLoader over repeated toy-corpus sequences (no HF dependency).

    With ``repeat=True`` (default) the loader yields batches forever by
    cycling, so a training loop can run an exact number of ``--steps``
    regardless of ``n_sequences``.
    """
    import itertools

    ds = ToyDataset(text=text, n_sequences=n_sequences)
    base = DataLoader(ds, batch_size=batch_size, shuffle=False)
    if not repeat:
        return base

    class _CyclicLoader:
        """Wrap a DataLoader so ``iter()`` never exhausts."""

        def __iter__(self):
            return itertools.cycle(base)

    return _CyclicLoader()  # type: ignore[return-value]


def tokenise_corpus(text: str | None = None, max_len: int = 256) -> tuple[Tensor, CharTokenizer]:
    tok = CharTokenizer(text if text is not None else DEFAULT_CORPUS)
    ids = [tok.bos_id] + tok.encode(text if text is not None else DEFAULT_CORPUS) + [tok.eos_id]
    ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0), tok


# ---------------------------------------------------------------------------
# HuggingFace dataset pipeline (for actual pre-training)
# ---------------------------------------------------------------------------

def _get_tokenizer(cfg: TrainingConfig):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloader(
    cfg: TrainingConfig,
    is_eval: bool = False,
) -> DataLoader:
    """Create a DataLoader for pre-training.

    Uses HuggingFace ``datasets`` (streaming by default) and tokenizes
    on-the-fly.  If ``cfg.dataset_streaming`` is False it caches the
    full dataset locally first.
    """
    try:
        import datasets as hf_datasets
    except ImportError:
        raise ImportError("pip install datasets to use the data pipeline")

    tokenizer = _get_tokenizer(cfg)

    def tokenize_fn(examples):
        texts = examples.get("text", examples.get("content", ["<unk>" * cfg.seq_length]))
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, truncation=False)
            if len(ids) == 0:
                ids = [tokenizer.eos_token_id or 0]
            all_ids.extend(ids + [tokenizer.eos_token_id or 0])
        input_ids = []
        for i in range(0, len(all_ids), cfg.seq_length):
            chunk = all_ids[i: i + cfg.seq_length]
            if len(chunk) == cfg.seq_length:
                input_ids.append(chunk)
        return {"input_ids": input_ids, "labels": input_ids}

    if cfg.dataset_streaming:
        split_spec = f"train[:{cfg.dataset_max_samples}]" if cfg.dataset_max_samples > 0 else "train"
        data = hf_datasets.load_dataset(
            cfg.dataset_name, cfg.dataset_subset, split=split_spec, streaming=True,
        )
    else:
        data = hf_datasets.load_dataset(
            cfg.dataset_name, cfg.dataset_subset, split="train",
            streaming=False, cache_dir=cfg.dataset_cache_dir,
        )
        if cfg.dataset_max_samples > 0:
            data = data.select(range(cfg.dataset_max_samples))

    data = data.map(
        tokenize_fn, batched=True, remove_columns=data.column_names, batch_size=100,
    )
    return DataLoader(
        data, batch_size=cfg.batch_size,
        num_workers=0 if cfg.dataset_streaming else 2, pin_memory=True,
    )
