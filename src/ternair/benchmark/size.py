"""Model size estimator for ternary (BitNet b1.58) models.

Compute the projected on-disk footprint from a :class:`TernairConfig`
itself, without instantiating the model. Useful as a notebook
pre-flight check.

Storage accounting (per linear)
------------------------------

For a ternary linear ``nn.Linear(out_features, in_features)``:

* The internal weight has ``out_features × in_features`` ternary
  parameters. After per-row scaling ``γ = mean(|W|)`` the weight
  tensor is quantised to ``{-1, 0, 1}``.
* In ``"packed"`` storage we keep 5 trits per byte (1.6 bits/value).
* In ``"int8"`` storage we keep 1 trit per byte (8 bits/value).
* One FP32 ``γ`` per *output row* (not per weight) - so ``out_features``
  scalars. This dominates γ only for tiny hidden sizes.

Embedding and LM head
---------------------

* Embeddings stay in their natural dtype (``embedding_dtype_bytes``
  defaults to 2 = FP16 for the demo but can be tightened to 1 for INT8).
* The LM head is shared with the embedding by default
  (``TernairConfig.tie_word_embeddings=True``), so its cost is 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from ternair.model.config import TernairConfig


@dataclass
class SizeBreakdown:
    ternary_linear_weights_bytes: int
    ternary_linear_scales_bytes: int
    embedding_bytes: int
    lm_head_bytes: int
    other_bytes: int  # RMSNorm γ, embedding scale, … (FP32)
    total_bytes: int
    ternary_param_count: int

    bits_per_value_avg: float
    bits_per_param_avg: float  # bytes-weighted including embedding overhead

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024 ** 2)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_gib"] = self.total_gib
        d["total_mib"] = self.total_mib
        return d


# ---------------------------------------------------------------------------
# Pure-Python formulas. We do not import torch here so this module is
# safe to use in any pure-Python context (e.g. notebook pre-flight).
# ---------------------------------------------------------------------------
def _ternary_params_per_layer(config: TernairConfig) -> int:
    """Total ternary parameters contributed by one decoder block."""
    H = config.hidden_size
    I = config.intermediate_size
    KVH = config.num_key_value_heads * config.head_dim
    return 2 * H * H + 2 * H * KVH + 3 * H * I


def _outputs_per_layer(config: TernairConfig) -> int:
    """Total ``out_features`` across all linears of one decoder block.

    ``γ`` is one FP32 scalar per output row, so the total number of
    γ scales per layer is just the sum of ``out_features``.
    """
    H = config.hidden_size
    I = config.intermediate_size
    KVH = config.num_key_value_heads * config.head_dim
    # Q: out=H ; K/V: out=KVH ; O: out=H ; gate/up: out=I ; down: out=H
    return 3 * H + 2 * KVH + 2 * I


def model_size_bytes(
    config: TernairConfig,
    embedding_dtype_bytes: int = 2,
    include_lm_head: Optional[bool] = None,
) -> SizeBreakdown:
    """Project the on-disk footprint of ``config``.

    Parameters
    ----------
    embedding_dtype_bytes:
        Set to 1 for INT8 embeddings, 2 for FP16 (default), 4 for FP32.
    include_lm_head:
        Defaults to ``not config.tie_word_embeddings``.
    """
    if config.storage == "packed":
        bits_per_value = 1.6
    elif config.storage == "fastpacked":
        bits_per_value = 2.0
    elif config.storage == "int8":
        bits_per_value = 8.0
    else:
        raise ValueError(f"Unsupported storage {config.storage!r}")

    per_layer = _ternary_params_per_layer(config)
    ternary_count = per_layer * config.num_hidden_layers

    ternary_weight_bytes = int(round(ternary_count * (bits_per_value / 8.0)))

    # γ is one scalar per output row, summed across the seven linears.
    outputs_per_layer = _outputs_per_layer(config)
    scale_bytes = outputs_per_layer * config.num_hidden_layers * 4

    embedding_bytes = config.vocab_size * config.hidden_size * embedding_dtype_bytes

    if include_lm_head is None:
        include_lm_head = not config.tie_word_embeddings
    lm_head_bytes = (
        config.hidden_size * config.vocab_size * embedding_dtype_bytes
        if include_lm_head
        else 0
    )

    # RMSNorm γ weights: 1×H FP32 per layer + final norm. Negligible.
    other_bytes = (config.num_hidden_layers + 1) * config.hidden_size * 4

    total = (
        ternary_weight_bytes
        + scale_bytes
        + embedding_bytes
        + lm_head_bytes
        + other_bytes
    )

    bits_per_param = (total * 8.0) / max(
        ternary_count + config.vocab_size * config.hidden_size, 1
    )

    return SizeBreakdown(
        ternary_linear_weights_bytes=ternary_weight_bytes,
        ternary_linear_scales_bytes=scale_bytes,
        embedding_bytes=embedding_bytes,
        lm_head_bytes=lm_head_bytes,
        other_bytes=other_bytes,
        total_bytes=total,
        ternary_param_count=ternary_count,
        bits_per_value_avg=bits_per_value,
        bits_per_param_avg=bits_per_param,
    )


def describe(config: TernairConfig, **kwargs) -> str:
    """Pretty summary of the projected size."""
    b = model_size_bytes(config, **kwargs)
    return (
        "ternair config size projection\n"
        "------------------------------\n"
        f"  storage              : {config.storage}  (~{b.bits_per_value_avg:.2f} bits/value)\n"
        f"  hidden_size          : {config.hidden_size}\n"
        f"  num_hidden_layers    : {config.num_hidden_layers}\n"
        f"  ternary params       : {b.ternary_param_count:,}  "
        f"({b.ternary_param_count / 1e9:.3f} B)\n"
        f"  ternary weights      : {b.ternary_linear_weights_bytes / 1024 ** 2:8.1f} MiB\n"
        f"  ternary scales       : {b.ternary_linear_scales_bytes / 1024.0:12.1f} KiB\n"
        f"  embedding (cfg bytes): {b.embedding_bytes / 1024 ** 2:8.1f} MiB\n"
        f"  lm head              : {b.lm_head_bytes / 1024.0:12.1f} KiB\n"
        f"  RMSNorm + buffer     : {b.other_bytes:8d} B\n"
        f"  TOTAL                : {b.total_bytes / 1024 ** 2:8.1f} MiB"
        f"  ({b.total_gib:.3f} GiB)\n"
        f"  effective bits/param : {b.bits_per_param_avg:.3f}\n"
    )


def auto_fit_to_bytes(
    base: TernairConfig,
    target_bytes: int,
    min_layers: int = 4,
    max_layers: int = 256,
) -> TernairConfig:
    """Tune ``num_hidden_layers`` so ``base`` fits within ``target_bytes``.

    Bytes scale linearly with ``num_hidden_layers`` for a fixed
    architecture, so we solve the budget directly rather than
    searching. The total is::

        total = fixed_bytes + per_layer_bytes * num_hidden_layers

    where ``per_layer_bytes`` covers weights + γ scales + RMSNorm, and
    ``fixed_bytes`` covers the embedding and the final norm.
    """
    per_layer_params = _ternary_params_per_layer(base)
    per_layer_outputs = _outputs_per_layer(base)
    bits_per_value = 1.6 if base.storage == "packed" else 8.0

    per_layer_bytes = (
        per_layer_params * (bits_per_value / 8.0)
        + per_layer_outputs * 4  # γ scales (FP32)
        + base.hidden_size * 4  # 1 RMSNorm (FP32) per layer
    )
    if per_layer_bytes <= 0:
        raise ValueError("Architecture has no ternary contributions to size")

    fixed_bytes = (
        base.vocab_size * base.hidden_size * 2  # FP16 embedding
        + base.hidden_size * 4  # final RMSNorm
    )

    budget = max(target_bytes - fixed_bytes, 0)
    needed = int(round(budget / per_layer_bytes))
    needed = max(min_layers, min(needed, max_layers))

    return TernairConfig(
        vocab_size=base.vocab_size,
        hidden_size=base.hidden_size,
        intermediate_size=base.intermediate_size,
        num_hidden_layers=needed,
        num_attention_heads=base.num_attention_heads,
        num_key_value_heads=base.num_key_value_heads,
        max_position_embeddings=base.max_position_embeddings,
        rope_theta=base.rope_theta,
        rms_norm_eps=base.rms_norm_eps,
        tie_word_embeddings=base.tie_word_embeddings,
        storage=base.storage,
    )


def fit_one_gb(base: TernairConfig) -> TernairConfig:
    """Tune ``base`` so its projected packed footprint stays under 1 GiB."""
    # Aim for 950 MiB so the safety margin bites before the ceiling.
    return auto_fit_to_bytes(base, target_bytes=int(1024 ** 2 * 950))


__all__ = [
    "SizeBreakdown",
    "describe",
    "model_size_bytes",
    "auto_fit_to_bytes",
    "fit_one_gb",
]
