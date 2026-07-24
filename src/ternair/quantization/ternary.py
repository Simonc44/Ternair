"""Ternary weight quantization — BitNet b1.58 style.

Forward:   W_t = round(clamp(W / γ, -1, 1))  ∈  {-1, 0, +1}
Backward:  STE — gradient flows through ``round`` as identity.

Quantization Annealing
----------------------
During QAT, a temperature ``beta`` controls the softness of the
continuous-to-ternary transition:

    beta  ≈  1   →  smooth tanh approximation  (early training)
    beta  →  ∞   →  hard round()               (end of training)

Key fix vs. previous version
-----------------------------
**Per-model state** via :class:`AnnealingState`.  The old
``_global_beta`` module-level float was shared across *all* models in
the same Python process, breaking multi-model training and
multi-threaded evaluation.  Each model / optimizer now owns its own
:class:`AnnealingState`, while a process-level default is kept for
backward compatibility with code that calls the bare
``set_quant_annealing_beta()`` API.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# AnnealingState — per-model, thread-safe beta holder
# ---------------------------------------------------------------------------

@dataclass
class AnnealingState:
    """Holds the current quantization-annealing temperature for one model.

    Parameters
    ----------
    beta_start:
        Initial temperature (smooth tanh, should be 1.0).
    beta_end:
        Final temperature (approximates hard round(), typically 10–15).

    Usage
    -----
    .. code-block:: python

        state = AnnealingState()
        optimizer = ...  # your optimizer

        for step in range(total_steps):
            beta = state.step(step, total_steps)
            # beta is automatically set on the state; ternary_linear_forward
            # accepts it as an explicit argument to bypass the global default.
    """

    beta_start: float = 1.0
    beta_end: float = 15.0
    warmup_ratio: float = 0.1
    _beta: float = field(default=1.0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def beta(self) -> float:
        with self._lock:
            return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        with self._lock:
            self._beta = max(1.0, float(value))

    def step(self, current_step: int, total_steps: int) -> float:
        """Advance the schedule and return the new beta."""
        beta = get_annealing_schedule(
            current_step,
            total_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            warmup_ratio=self.warmup_ratio,
        )
        self.beta = beta
        return beta

    def reset(self) -> None:
        """Reset beta to beta_start (useful between training runs)."""
        self.beta = self.beta_start


def create_annealing_state(
    beta_start: float = 1.0,
    beta_end: float = 15.0,
    warmup_ratio: float = 0.1,
) -> AnnealingState:
    """Factory for a per-model :class:`AnnealingState`.

    Prefer this over the module-level ``set_quant_annealing_beta()`` when
    training multiple models in the same process.
    """
    return AnnealingState(
        beta_start=beta_start,
        beta_end=beta_end,
        warmup_ratio=warmup_ratio,
    )


# ---------------------------------------------------------------------------
# Process-level default — backward-compatible API
# ---------------------------------------------------------------------------

_default_state = AnnealingState()


def set_quant_annealing_beta(beta: float) -> None:
    """Set the process-level default annealing temperature.

    .. deprecated::
        Prefer :func:`create_annealing_state` for per-model isolation.
    """
    _default_state.beta = beta


def get_quant_annealing_beta() -> float:
    """Return the current process-level default beta."""
    return _default_state.beta


def get_annealing_schedule(
    current_step: int,
    total_steps: int,
    beta_start: float = 1.0,
    beta_end: float = 15.0,
    warmup_ratio: float = 0.1,
) -> float:
    """Compute beta for a given step using a linear warmup-then-anneal schedule.

    Parameters
    ----------
    current_step:
        Current training step (0-indexed).
    total_steps:
        Total number of training steps.
    beta_start:
        Beta at the start (after warmup).
    beta_end:
        Beta at the end of training.
    warmup_ratio:
        Fraction of total_steps held at beta_start.

    Returns
    -------
    float
        The beta value for this step.
    """
    if total_steps <= 0:
        return beta_end

    warmup_steps = int(total_steps * warmup_ratio)
    if current_step < warmup_steps:
        return beta_start

    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
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
    """Per-row scale γ = mean(|W|) along ``dim`` (BitNet b1.58 reference)."""
    return w.abs().mean(dim=dim, keepdim=True).clamp_min(1e-8)


def ternarize(w: Tensor, dim: int = -1) -> tuple[Tensor, Tensor]:
    """Inference-mode ternary quantization — no gradient flow.

    Returns ``(trits, gamma)`` with trits in ``{-1, 0, +1}`` (int8)
    and gamma in float32.
    """
    gamma = _compute_gamma(w, dim=dim)
    w_norm = w / gamma
    w_clip = torch.clamp(w_norm, -1.0, 1.0)
    w_ternary = torch.round(w_clip)
    return w_ternary.to(torch.int8), gamma.to(torch.float32)


def ternarize_ste(
    w: Tensor,
    dim: int = -1,
    use_annealing: bool = True,
    annealing_state: AnnealingState | None = None,
) -> tuple[Tensor, Tensor]:
    """Training-mode ternary quantization with straight-through estimator.

    Parameters
    ----------
    w:
        FP weight tensor to ternarise.
    dim:
        Dimension along which to compute per-row gamma.
    use_annealing:
        If True, use the tanh approximation when beta > 1.
    annealing_state:
        Per-model :class:`AnnealingState`.  Falls back to the
        process-level default if None.

    Returns
    -------
    (w_t_effective, gamma)
        ``w_t_effective = gamma * w_t`` with STE in the backward pass.
    """
    state = annealing_state if annealing_state is not None else _default_state
    beta = state.beta

    gamma = _compute_gamma(w, dim=dim)
    w_norm = w / gamma

    if use_annealing and beta > 1.0:
        w_tanh = torch.tanh(beta * w_norm)
        w_effective = w_tanh + (w_norm - w_norm.detach())  # STE
        return (gamma * w_effective).to(w.dtype), gamma.to(torch.float32)

    w_clip = torch.clamp(w_norm, -1.0, 1.0)
    w_ternary = torch.round(w_clip)
    w_effective = w_ternary + (w_norm - w_norm.detach())  # STE
    return (gamma * w_effective).to(w.dtype), gamma.to(torch.float32)


def stats_from(w_t: Tensor, gamma: Tensor) -> TernaryStats:
    """Summarise the content of a ternarised tensor."""
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

    Forward:  gamma * round(clamp(W/gamma, -1, 1))  (or tanh approx)
    Backward: pass gradient through as identity on W; zero for gamma.
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
    annealing_state: AnnealingState | None = None,
) -> Tensor:
    """Apply the ternary forward in an autograd-friendly way.

    Parameters
    ----------
    w, gamma:
        Weight tensor and its scale.
    beta:
        Explicit temperature override.  If None, reads from
        ``annealing_state`` (or the process-level default).
    annealing_state:
        Per-model :class:`AnnealingState` for isolation.
    """
    if beta is None:
        state = annealing_state if annealing_state is not None else _default_state
        beta = state.beta
    return TernaryLinearFn.apply(w, gamma, beta)


__all__ = [
    "AnnealingState",
    "create_annealing_state",
    "TernaryStats",
    "set_quant_annealing_beta",
    "get_quant_annealing_beta",
    "get_annealing_schedule",
    "ternarize",
    "ternarize_ste",
    "stats_from",
    "ternary_linear_forward",
]

# Suppress unused-import lint for nn (reserved for future SymmetricQuantize)
_ = nn
