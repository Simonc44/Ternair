"""Ready-made model profiles that fit a given memory budget.

The 1 GiB profile is engineered so that the same model is bounded by
the 1 GB ceiling when using packed storage; we expose its size
calculator output in :func:`ternair.benchmark.size.describe`.
"""

from __future__ import annotations

from ternair.model.config import TernairConfig


def tiny_profile(storage: str = "packed") -> TernairConfig:
    """A 25 M-parameter toy model that fits in ~6 MiB packed (~25 MiB int8)."""
    return TernairConfig(
        vocab_size=4096,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        storage=storage,
    )


def base_profile(storage: str = "packed") -> TernairConfig:
    """~700 M-parameter profile, ~150 MiB packed."""
    return TernairConfig(
        vocab_size=32768,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=18,
        num_attention_heads=24,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        storage=storage,
    )


def one_gb_profile(storage: str = "packed") -> TernairConfig:
    """Profile engineered to land just under 1 GiB once packed (≈ 970 MiB).

    60 transformer blocks × ≈ 67.83 M ternary params/block = ≈ 4.07 B.
    Packed at 1.6 bits/value that is ≈ 813 MiB, plus a 32 768×2560 FP16
    embedding (≈ 160 MiB) gives ≈ 973 MiB - very close to the 1 GiB
    ceiling. Use :func:`ternair.benchmark.fit_one_gb` to auto-tune
    if the actual measurement falls short on a given architecture.
    """
    return TernairConfig(
        vocab_size=32768,
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=60,
        num_attention_heads=32,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        storage=storage,
    )


__all__ = ["tiny_profile", "base_profile", "one_gb_profile"]
