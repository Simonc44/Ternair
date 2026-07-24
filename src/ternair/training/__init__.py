"""Tiny training helpers (smoke tests only — not a production trainer)."""

from ternair.training.data import toy_corpus, tokenise_corpus
from ternair.training.trainer import train_one_step, cross_entropy

__all__ = ["toy_corpus", "tokenise_corpus", "train_one_step", "cross_entropy"]
