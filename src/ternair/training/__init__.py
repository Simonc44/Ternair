"""Tiny training helpers (smoke tests only -- not a production trainer)."""

from ternair.training.atomic import (
    AtomicCheckpointSaver,
    DEFAULT_FILENAME,
    PREVIOUS_FILENAME,
)
from ternair.training.data import toy_corpus, tokenise_corpus
from ternair.training.memory import (
    DEFAULT_BYTES_PER_PARAM_MASTER,
    DEFAULT_BYTES_PER_PARAM_OPTIM,
    MemoryEstimate,
    estimate_activations_bytes,
    estimate_memory,
)
from ternair.training.pipeline import (
    PipelineStage,
    PipelineState,
    TernairPipeline,
)
from ternair.training.trainer import train_one_step, cross_entropy

__all__ = [
    "toy_corpus",
    "tokenise_corpus",
    "train_one_step",
    "cross_entropy",
    # v0.5.0 additions for intermediate-size robustness
    "AtomicCheckpointSaver",
    "DEFAULT_FILENAME",
    "PREVIOUS_FILENAME",
    "MemoryEstimate",
    "estimate_memory",
    "estimate_activations_bytes",
    "DEFAULT_BYTES_PER_PARAM_MASTER",
    "DEFAULT_BYTES_PER_PARAM_OPTIM",
    "TernairPipeline",
    "PipelineStage",
    "PipelineState",
]
