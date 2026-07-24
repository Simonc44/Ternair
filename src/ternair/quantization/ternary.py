"""Ternary weight quantization - BitNet b1.58 style.

Forward pass stores ``W_t = round(clamp(W / γ, -1, 1)) ∈ {-1, 0, +1}``.
Backward pass uses STE so that the gradient flows through ``round`` as
if it were the identity.

Integrates Quantization Annealing (recuit de quantification) avec un
parametre de temperature ``beta`` qui controle la douceur de la
transition continu -> ternaire pendant le QAT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Annealing state (temperature beta globale)
# ---------------------------------------------------------------------------

# Temperature de quantification pour le recuit (Quantization Annealing)
# beta = 1.0 -> tres lisse (debut d'entrainement)
# beta = 10.0+ -> proche de round() (fin d'entrainement)
_global_beta: float = 1.0


def set_quant_annealing_beta(beta: float) -> None:
    """Modifie la temperature beta pour le recuit de quantification.
    
    Pendant le QAT, on augmente progressivement beta de 1.0 a 10.0+
    pour passer d'une quantification douce (tanh) a une quantification
    dure (round).
    
    Args:
        beta: Temperature de quantification (>= 1.0)
    """
    global _global_beta
    _global_beta = max(1.0, beta)


def get_quant_annealing_beta() -> float:
    """Retourne la temperature beta actuelle."""
    global _global_beta
    return _global_beta


def get_annealing_schedule(
    current_step: int,
    total_steps: int,
    beta_start: float = 1.0,
    beta_end: float = 15.0,
    warmup_ratio: float = 0.1,
) -> float:
    """Calcule beta selon un schedule lineaire de recuit.
    
    Args:
        current_step: Etape actuelle
        total_steps: Nombre total d'etapes
        beta_start: Valeur initiale de beta (douce)
        beta_end: Valeur finale de beta (dure)
        warmup_ratio: Proportion d'etapes en warmup (beta stable)
    
    Returns:
        Valeur de beta pour l'etape courante
    """
    if total_steps <= 0:
        return beta_end
    
    warmup_steps = int(total_steps * warmup_ratio)
    
    if current_step < warmup_steps:
        return beta_start
    
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    
    # Croissance lineaire de beta_start a beta_end
    return beta_start + (beta_end - beta_start) * progress


# ---------------------------------------------------------------------------
# Gamma computation
# ---------------------------------------------------------------------------


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


def _tanh_ternarize(w: Tensor, scale: Tensor, beta: float) -> Tensor:
    """Ternarisation douce via tanh pour le recuit de quantification.
    
    Au lieu de round(clamp(x, -1, 1)), on utilise :
        W_proxy = tanh(beta * W / alpha) * alpha
    
    Quand beta -> inf, tanh(beta * x) -> sign(x) = {-1, 0, +1} approxime.
    Quand beta est proche de 1, tanh donne une approximation differentiable.
    """
    w_norm = w / scale
    # tanh beta * w_norm donne une approximation douce de sign()
    w_tanh = torch.tanh(beta * w_norm)
    return w_tanh * scale


def ternarize_ste(
    w: Tensor,
    dim: int = -1,
    use_annealing: bool = True,
) -> tuple[Tensor, Tensor]:
    """Training-mode ternary quantization with straight-through estimator.

    Si ``use_annealing`` est True, utilise la temperature beta globale
    pour un recuit progressif (tanh -> round) pendant le QAT.

    Returns ``(w_t_effective, γ)`` where ``w_t_effective = γ · w_t``,
    the gradient of which is treated as ``γ`` (i.e. the STE forwards
    ``w_t`` but the backward pass replaces it with the identity on ``w``).
    """
    gamma = _compute_gamma(w, dim=dim)
    w_norm = w / gamma
    
    if use_annealing and _global_beta > 1.0:
        # Recuit : approximation differentiable via tanh
        beta = _global_beta
        w_tanh = torch.tanh(beta * w_norm)
        w_effective = w_tanh + (w_norm - w_norm.detach())  # STE
        return (gamma * w_effective).to(w.dtype), gamma.to(torch.float32)
    
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
    def forward(ctx, w: Tensor, gamma: Tensor, beta: float = 1.0) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(w)
        ctx.beta = beta
        w_norm = w / gamma
        if beta > 1.0:
            w_t = torch.tanh(beta * w_norm)
        else:
            w_clip = torch.clamp(w_norm, -1.0, 1.0)
            w_t = torch.round(w_clip)
        return gamma * w_t

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        (w,) = ctx.saved_tensors
        return grad_out, None, None


def ternary_linear_forward(
    w: Tensor,
    gamma: Tensor,
    beta: float | None = None,
) -> Tensor:
    """Apply the ternary forward in a way that's autograd-friendly.
    
    Args:
        w: Poids a ternariser
        gamma: Facteur d'echelle calcule
        beta: Temperature de recuit (None = utiliser globale)
    """
    if beta is None:
        beta = _global_beta
    return TernaryLinearFn.apply(w, gamma, beta)


__all__ = [
    "TernaryStats",
    "set_quant_annealing_beta",
    "get_quant_annealing_beta",
    "get_annealing_schedule",
    "ternarize",
    "ternarize_ste",
    "stats_from",
    "ternary_linear_forward",
]


# Avoid "unused" lint for nn (we may bring nn.SymmetricQuantize later)
_ = nn
