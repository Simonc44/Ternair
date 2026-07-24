"""8-bit per-token absmax activation quantization (BitNet b1.58).

During training and inference, hidden activations are quantised to
``int8`` using ``γ_a = max(|x|) / 127`` per token (last dimension).
The forward uses STE; the backward treats the quantisation as identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Activation8Bit:
    quantised: Tensor  # int8, same shape as input
    scale: Tensor  # fp32, shape broadcastable to (..., 1)


class _ActivationQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # type: ignore[override]
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        scale = absmax / 127.0
        x_int = torch.clamp(torch.round(x / scale), -128.0, 127.0)
        # Per-token quantisation produces a non-differentiable function;
        # pass gradient through as if it were identity (STE).
        return x_int.to(torch.float32) * scale

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        return grad_out


def quantize_activations_8bit(x: Tensor) -> Activation8Bit:
    """Quantise ``x`` per-token to ``int8`` using absmax.

    Returns both the quantized values (int8) and the per-token
    scale so they can be inspected or recombined outside the STE
    function.
    """
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(x / scale), -128.0, 127.0).to(torch.int8)
    return Activation8Bit(quantised=q, scale=scale.detach().to(torch.float32))


def quantize_activations_8bit_forward(x: Tensor) -> Tensor:
    """Forward pass for activations with STE for backprop."""
    return _ActivationQuantFn.apply(x)


__all__ = [
    "Activation8Bit",
    "quantize_activations_8bit",
    "quantize_activations_8bit_forward",
]
