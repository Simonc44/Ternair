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
from ternair.quantization.packing import pack_trits, unpack_trits

__all__ = [
    "TernaryStats",
    "ternarize",
    "ternarize_ste",
    "quantize_activations_8bit",
    "Activation8Bit",
    "TernairLinear",
    "TernairLinearStorage",
    "pack_trits",
    "unpack_trits",
]
