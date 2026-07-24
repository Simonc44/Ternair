"""Ternary weight quantization - BitNet b1.58 style.

Forward pass stores ``W_t = round(clamp(W / γ, -1, 1)) ∈ {-1, 0, +1}``.
Backward pass uses STE so that the gradient flows through ``round`` as
if it were the identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class TernaryStats:
    """Quick diagnostics of a ternarised tensor."""

    numel: int
    num_pos: int
    num_zero: int
    num_neg: int
    gamma: float
    sparsity: float  # fraction of zero entries


def _compute_gamma(w: Tensor, dim: int = -1) -> Tensor:
    """Per-row γ = mean(|W|) along ``dim``.

    The original BitNet b1.58 paper uses ``mean(|W|)`` per output row; we
    keep a separate γ per output channel so that each filter is scaled
    independently. This mirrors the official reference implementation.
    """
    return w.abs().mean(dim=dim, keepdim=True).clamp_min(1e-8)


def ternarize(w: Tensor, dim: int = -1) -> tuple[Tensor, Tensor]:
    """Inference-mode ternary quantization.

    Quantises ``w`` along ``dim`` to ``{-1, 0, +1}`` and returns the
    scale ``γ``. No gradient flow.
    """
    gamma = _compute_gamma(w, dim=dim)
    w_norm = w / gamma
    w_clip = torch.clamp(w_norm, -1.0, 1.0)
    w_ternary = torch.round(w_clip)
    return w_ternary.to(torch.int8), gamma.to(torch.float32)


def ternarize_ste(w: Tensor, dim: int = -1) -> tuple[Tensor, Tensor]:
    """Training-mode ternary quantization with straight-through estimator.

    Returns ``(w_t_effective, γ)`` where ``w_t_effective = γ · w_t``,
    the gradient of which is treated as ``γ`` (i.e. the STE forwards
    ``w_t`` but the backward pass replaces it with the identity on ``w``).
    """
    gamma = _compute_gamma(w, dim=dim)
    w_norm = w / gamma
    w_clip = torch.clamp(w_norm, -1.0, 1.0)
    w_ternary = torch.round(w_clip)
    # Forward uses γ · w_t, backward treats it as γ · w (STE).
    w_effective = w_ternary + (w_norm - w_norm.detach())
    return (gamma * w_effective).to(w.dtype), gamma.to(torch.float32)


def stats_from(w_t: Tensor, gamma: Tensor) -> TernaryStats:
    """Summarise the content of a ternarised tensor for diagnostics."""
    flat_t = w_t.detach().reshape(-1).to(torch.int32)
    numel = int(flat_t.numel())
    num_pos = int((flat_t == 1).sum().item())
    num_zero = int((flat_t == 0).sum().item())
    num_neg = int((flat_t == -1).sum().item())
    return TernaryStats(
        numel=numel,
        num_pos=num_pos,
        num_zero=num_zero,
        num_neg=num_neg,
        gamma=float(gamma.detach().mean().item()),
        sparsity=num_zero / max(numel, 1),
    )


class TernaryLinearFn(torch.autograd.Function):
    """Custom autograd function for ternary weights (STE).

    Forward: return ``γ · round(clamp(W/γ, -1, 1))``.
    Backward: pass through gradient as if the operation were identity
    on the FP weights, but zero out the gradient w.r.t. ``γ`` (it has no
    useful gradient in this simplified setting).
    """

    @staticmethod
    def forward(ctx, w: Tensor, gamma: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(w)
        w_norm = w / gamma
        w_clip = torch.clamp(w_norm, -1.0, 1.0)
        w_t = torch.round(w_clip)
        return gamma * w_t

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        (w,) = ctx.saved_tensors
        return grad_out, None


def ternary_linear_forward(w: Tensor, gamma: Tensor) -> Tensor:
    """Apply the ternary forward in a way that's autograd-friendly."""
    return TernaryLinearFn.apply(w, gamma)


__all__ = [
    "TernaryStats",
    "ternarize",
    "ternarize_ste",
    "stats_from",
    "ternary_linear_forward",
]


# Avoid "unused" lint for nn (we may bring nn.SymmetricQuantize later)
_ = nn
