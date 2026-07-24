"""ternair — a stand-alone BitNet-b1.58 ternary language model prototype.

The package implements the BitNet b1.58 recipe:

* Per-row ternary weight quantisation ``{-1, 0, +1}`` with a
  per-output scale ``γ`` (see :mod:`ternair.quantization.ternary`).
* Per-token 8-bit absmax activation quantisation
  (see :mod:`ternair.quantization.activation`).
* A straight-through estimator (STE) for the ternary forward so the
  stack trains with normal back-prop.
* Decoder-only transformer blocks (RMSNorm + GQA RoPE attention +
  squared-ReLU MLP).
* A size estimator that quantifies the gain of the 1.58-bit
  representation: e.g. the high-level profile fits under 1 GiB total
  while holding ≈ 3-4 B ternary weights.

Entry points
------------

* :mod:`ternair.model` — :class:`TernairForCausalLM`,
  :class:`TernairConfig`, ready-made :func:`tiny_profile`,
  :func:`base_profile`, :func:`one_gb_profile`.
* :mod:`ternair.quantization` — bit-level utilities.
* :mod:`ternair.benchmark` — :func:`describe` / :func:`model_size_bytes`.
* :mod:`ternair.training` — :func:`train_one_step` smoke test.
* CLI via ``python -m ternair …`` — see :mod:`ternair.cli`.
"""

from ternair._version import __version__
from ternair.quantization import (
    Activation8Bit,
    TernairLinear,
    TernairLinearStorage,
    TernaryStats,
    pack_trits,
    quantize_activations_8bit,
    ternarize,
    ternarize_ste,
    unpack_trits,
)
from ternair.model import (
    RMSNorm,
    TernairAttention,
    TernairBlock,
    TernairConfig,
    TernairForCausalLM,
    TernairModel,
    TernairMLP,
    base_profile,
    generate,
    one_gb_profile,
    tiny_profile,
)
from ternair.benchmark.size import (
    SizeBreakdown,
    auto_fit_to_bytes,
    describe as describe_size,
    fit_one_gb,
    model_size_bytes,
)
from ternair.kernels import MODE_FASTPACKED
from ternair.kernels.packing_fast import pack_trits_2bit, unpack_trits_2bit
from ternair.kernels.packing_base8 import (
    MODE_BASE8,
    MODE_PACKED,
    BITS_PER_VALUE,
    pack_trits_base8,
    unpack_trits_base8,
)
from ternair.kernels.triton_fast import (
    has_triton,
    ternary_matmul_triton,
)
from ternair.training import cross_entropy, tokenise_corpus, train_one_step


__all__ = [
    "__version__",
    # quantization
    "Activation8Bit",
    "TernairLinear",
    "TernairLinearStorage",
    "TernaryStats",
    "pack_trits",
    "pack_trits_base8",
    "quantize_activations_8bit",
    "ternarize",
    "ternarize_ste",
    "unpack_trits",
    "unpack_trits_base8",
    # model
    "RMSNorm",
    "TernairAttention",
    "TernairBlock",
    "TernairConfig",
    "TernairForCausalLM",
    "TernairModel",
    "TernairMLP",
    "base_profile",
    "generate",
    "one_gb_profile",
    "tiny_profile",
    # benchmark
    "SizeBreakdown",
    "auto_fit_to_bytes",
    "describe_size",
    "fit_one_gb",
    "model_size_bytes",
    # kernels
    "MODE_FASTPACKED",
    "MODE_BASE8",
    "MODE_PACKED",
    "BITS_PER_VALUE",
    "pack_trits_2bit",
    "unpack_trits_2bit",
    "has_triton",
    "ternary_matmul_triton",
    # training
    "cross_entropy",
    "tokenise_corpus",
    "train_one_step",
]
