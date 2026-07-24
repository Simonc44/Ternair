"""Ternary weight + 8-bit activation quantization primitives.

This subpackage implements the BitNet b1.58 / BitNet-style scheme:

* Linear weights are quantised to ``{-1, 0, +1}`` with a per-output
  scalar ``γ`` so that ``W ≈ γ · W_t`` with ``W_t`` ternary.
* Activations are quantised to 8-bit per-token using abs-max scaling,
  matching the original paper.
* A straight-through estimator (STE) is used during training so that
  gradients flow through the non-differentiable ``round()`` step.
"""

from ternair.quantization.ternary import TernaryStats, ternarize, ternarize_ste
from ternair.quantization.activation import quantize_activations_8bit, Activation8Bit
from ternair.quantization.linear import TernairLinear, TernairLinearStorage
from ternair.kernels.packing_base8 import (
    MODE_BASE8,
    MODE_PACKED,
    BITS_PER_VALUE,
    pack_trits,
    pack_trits_base8,
    unpack_trits,
    unpack_trits_base8,
)

__all__ = [
    "TernaryStats",
    "ternarize",
    "ternarize_ste",
    "quantize_activations_8bit",
    "Activation8Bit",
    "TernairLinear",
    "TernairLinearStorage",
    # Packing (v0.6.0 canonical location: kernels/packing_base8)
    "MODE_BASE8",
    "MODE_PACKED",
    "BITS_PER_VALUE",
    "pack_trits",
    "pack_trits_base8",
    "unpack_trits",
    "unpack_trits_base8",
]
