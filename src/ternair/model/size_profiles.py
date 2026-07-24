"""Ready-made model profiles that fit a given memory budget.

The profiles cover the spectrum from a CPU-runnable smoke-test model
(``tiny``) up to a 1 GiB ceiling (``one_gb``).  Intermediate profiles
(``small``, ``medium``, ``large``) close the 50M-500M gap and are the
recommended targets for researchers who want a model that actually
runs end-to-end on a single accelerator without being a toy.

Number of ternary params is approximate: ``≈ 2*H² + 3*H*I`` per
decoder block, where ``H`` is ``hidden_size`` and ``I`` is
``intermediate_size``.
"""

from __future__ import annotations

import math
from typing import Tuple

from ternair.benchmark.size import model_size_bytes
from ternair.model.config import TernairConfig


def tiny_profile(storage: str = "packed") -> TernairConfig:
    """~2.6 M-parameter toy model (~2.5 MiB packed).  CPU-runnable."""
    return TernairConfig(
        vocab_size=4096,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        storage=storage,
    )


def small_profile(storage: str = "packed") -> TernairConfig:
    """~50 M-parameter model (~10 MiB packed).

    First non-toy profile: 12 layers of width 768.  Trains on a T4 in
    a reasonable wall-clock, fits in BF16 + AdamW state under 4 GiB of
    VRAM.  Recommended for verifying distillation scripts.
    """
    return TernairConfig(
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        storage=storage,
    )


def medium_profile(storage: str = "packed") -> TernairConfig:
    """~150 M-parameter model (~30 MiB packed).

    18 layers of width 1024.  The sweet spot for intermediate-scale
    research: large enough to capture pattern-level statistics, small
    enough to train on a single A100 (40 GB) with a 1024-token context.
    """
    return TernairConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=18,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        storage=storage,
    )


def large_profile(storage: str = "packed") -> TernairConfig:
    """~250 M-parameter model (~55 MiB packed).

    24 layers of width 1280.  Targets large-recipe distillation: still
    under 100 MiB once packed, trains on a single A100 (80 GB) with
    full QAT in twelve hours.
    """
    return TernairConfig(
        vocab_size=32000,
        hidden_size=1280,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=20,
        num_key_value_heads=4,
        max_position_embeddings=2048,
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
    """Profile engineered to land just under 1 GiB once packed (~970 MiB).

    60 transformer blocks x ~67.83 M ternary params/block = ~4.07 B.
    Packed at 1.6 bits/value that is ~813 MiB, plus a 32 768x2560 FP16
    embedding (~160 MiB) gives ~973 MiB - very close to the 1 GiB
    ceiling.  Use :func:`ternair.benchmark.fit_one_gb` to auto-tune
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


def fit_profile_for_budget(
    target_mib: float,
    *,
    storage: str = "packed",
    min_hidden_size: int = 512,
    max_hidden_size: int = 2560,
    fixed_layers: int | None = None,
) -> Tuple[int, int]:
    """Pick ``(hidden_size, num_hidden_layers)`` that lands in the budget.

    ``target_mib`` is the packed-model budget (MiB).  We solve
    ``bytes = per_layer * L + fixed`` directly using the analytic
    formula in :func:`ternair.benchmark.size.model_size_bytes`.

    Returns a tuple ``(hidden_size, num_hidden_layers)`` that should
    land within +/- 5%% of ``target_mib``.  Useful when the user knows
    the storage budget but not the architecture.
    """
    from ternair.model.config import TernairConfig as _Cfg

    best: tuple[int, int] = (0, 0)
    best_err = float("inf")
    target_bytes = int(target_mib * 1024 ** 2)
    for h in range(min_hidden_size, max_hidden_size + 1, 128):
        cfg = _Cfg(
            hidden_size=h,
            intermediate_size=round(2.75 * h / 32) * 32,
            num_hidden_layers=fixed_layers or 12,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=1024,
            storage=storage,
        )
        b = model_size_bytes(cfg).total_bytes
        # Compute best L for that h.
        from ternair.benchmark.size import _ternary_params_per_layer, _outputs_per_layer
        per_layer_params = _ternary_params_per_layer(cfg)
        per_layer_outputs = _outputs_per_layer(cfg)
        bits_per_value = 1.6 if storage == "packed" else 2.0
        per_layer_bytes = int(per_layer_params * (bits_per_value / 8.0)) + per_layer_outputs * 4 + h * 4
        fixed_bytes = cfg.vocab_size * h * 2 + h * 4
        if per_layer_bytes <= 0:
            continue
        budget = max(target_bytes - fixed_bytes, 0)
        L = max(2, int(round(budget / per_layer_bytes)))
        cfg = _Cfg(
            hidden_size=h,
            intermediate_size=round(2.75 * h / 32) * 32,
            num_hidden_layers=L,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=1024,
            storage=storage,
        )
        b = model_size_bytes(cfg).total_bytes
        err = abs(b - target_bytes) / max(target_bytes, 1)
        if err < best_err:
            best_err = err
            best = (h, L)
    return best


PROFILE_REGISTRY: dict = {
    "tiny": tiny_profile,
    "small": small_profile,
    "base": base_profile,
    "medium": medium_profile,
    "large": large_profile,
    "one_gb": one_gb_profile,
}

__all__ = [
    "tiny_profile",
    "small_profile",
    "medium_profile",
    "large_profile",
    "base_profile",
    "one_gb_profile",
    "fit_profile_for_budget",
    "PROFILE_REGISTRY",
]
