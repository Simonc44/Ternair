"""8-bit per-token absmax activation quantization (BitNet b1.58).

During training and inference, hidden activations are quantised to
``int8`` using ``γ_a = max(|x|) / 127`` per token (last dimension).
The forward uses STE; the backward treats the quantisation as identity.

Hadamard transform (QuaRot/SpinQuant) is available but **disabled by
default during training from scratch**.  It is only useful at
calibration time (post-training quantization) when the model is
already trained in FP16 and we want to reduce INT8 outliers.
Enabling it during random-init training causes NaN cascades because
the butterfly loop amplifies initialisation noise before the weights
have had any chance to stabilise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Hadamard Transform (QuaRot / SpinQuant)
# Only use at calibration time, NOT during training from scratch.
# ---------------------------------------------------------------------------

def _hadamard_matrix(n: int, device: torch.device = None) -> Tensor:
    """Normalised Hadamard matrix of size n (n must be a power of 2)."""
    if n & (n - 1) != 0:
        raise ValueError(
            f"Size {n} must be a power of 2 for the Hadamard transform."
        )
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < n:
        h = torch.cat([
            torch.cat([h,  h], dim=1),
            torch.cat([h, -h], dim=1),
        ], dim=0) / (2.0 ** 0.5)
    return h


def apply_hadamard_transform(x: Tensor, dim: int = -1) -> Tensor:
    """Fast Walsh-Hadamard Transform (butterfly, O(n log n)).

    .. warning::
        Only call this at calibration / PTQ time.  During training from
        scratch the butterfly loop amplifies random-init noise and causes
        NaN cascades.  Use ``quantize_activations_8bit_forward`` with
        ``use_hadamard=False`` (the default) during training.
    """
    n = x.shape[dim]
    if n & (n - 1) != 0:
        raise ValueError(
            f"Dimension {dim} has size {n} which is not a power of 2."
        )
    x = x.transpose(dim, -1)
    shape = x.shape
    x = x.reshape(-1, n)
    h = 1
    while h < n:
        step = h * 2
        x_out = torch.empty_like(x)
        for i in range(0, n, step):
            u = x[:, i:i + h]
            v = x[:, i + h:i + step]
            x_out[:, i:i + h]       = (u + v) / (2.0 ** 0.5)
            x_out[:, i + h:i + step] = (u - v) / (2.0 ** 0.5)
        x = x_out
        h = step
    x = x.reshape(shape)
    return x.transpose(dim, -1)


def apply_inverse_hadamard(x: Tensor, dim: int = -1) -> Tensor:
    """Inverse FWHT (identical to forward since H^T = H)."""
    return apply_hadamard_transform(x, dim=dim)


# ---------------------------------------------------------------------------
# Activation 8-bit Quantisation
# ---------------------------------------------------------------------------

@dataclass
class Activation8Bit:
    quantised: Tensor  # int8, same shape as input
    scale: Tensor      # fp32, shape broadcastable to (..., 1)


class _ActivationQuantFn(torch.autograd.Function):
    """STE wrapper: forward quantises, backward passes gradient as-is."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # type: ignore[override]
        # Per-token absmax scale, clamped to avoid div-by-zero
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        scale  = absmax / 127.0
        x_int  = torch.clamp(torch.round(x / scale), -128.0, 127.0)
        return x_int * scale   # dequantised FP32

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        return grad_out


def quantize_activations_8bit(
    x: Tensor,
    use_hadamard: bool = False,   # disabled by default
) -> Activation8Bit:
    """Quantise ``x`` per-token to ``int8`` using absmax.

    Parameters
    ----------
    use_hadamard:
        Apply a Hadamard transform before quantisation to smooth outliers
        (QuaRot-style).  Only enable at **calibration time** after the
        model is already trained.  Enabling during random-init training
        causes NaN cascades.  Default: False.
    """
    if use_hadamard:
        x = apply_hadamard_transform(x, dim=-1)
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    scale  = absmax / 127.0
    q = torch.clamp(torch.round(x / scale), -128.0, 127.0).to(torch.int8)
    return Activation8Bit(quantised=q, scale=scale.detach().to(torch.float32))


def quantize_activations_8bit_forward(
    x: Tensor,
    use_hadamard: bool = False,   # disabled by default — see docstring
) -> Tensor:
    """Forward pass activation quantisation with STE for backprop.

    Parameters
    ----------
    use_hadamard:
        Apply Hadamard transform before/after quantisation.  Only safe
        at calibration time, not during training from scratch.
        Default: **False**.

    Notes
    -----
    Why is Hadamard disabled by default?
    The butterfly loop redistributes values across all channels.  With
    random initialisation this amplifies small-magnitude activations
    into large outliers in other channels, making the per-token absmax
    scale huge and collapsing almost every value to 0 after quantisation.
    After a few layers the hidden states are all-zero → loss = log(V)
    indefinitely → apparent NaN / constant PPL.

    Enable ``use_hadamard=True`` only after running
    :func:`calibrate_scale_equivalence` on a trained checkpoint.
    """
    if use_hadamard:
        x = apply_hadamard_transform(x, dim=-1)
        out = _ActivationQuantFn.apply(x)
        return apply_inverse_hadamard(out, dim=-1)
    return _ActivationQuantFn.apply(x)


# ---------------------------------------------------------------------------
# OmniQuant — learnable scale equivalence S
# ---------------------------------------------------------------------------

class ScaleEquivalence(nn.Module):
    """Learnable scale equivalence matrix S (OmniQuant-style).

    Optimises a diagonal per-channel scale factor to minimise
    quantisation error between FP16 and ternary + 8-bit outputs.

    Forward::
        x_s = x / scale
        W_s = scale.unsqueeze(-1) * W
        y = ternair_linear(x_s, W_s)
    """

    def __init__(self, hidden_size: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(
            torch.full((hidden_size,), math.log(init_scale))
        )

    @property
    def scale(self) -> Tensor:
        return torch.exp(self.log_scale)

    def apply_to_weights(self, weight: Tensor) -> Tensor:
        return weight * self.scale.unsqueeze(0)

    def apply_to_activations(self, x: Tensor) -> Tensor:
        return x / self.scale

    def forward(self, x: Tensor, weight: Tensor) -> Tensor:
        return self.apply_to_activations(x), self.apply_to_weights(weight)


def _compute_gamma(w: Tensor, dim: int = -1) -> Tensor:
    """Per-row scale gamma = mean(|W|) along dim."""
    return w.abs().mean(dim=dim, keepdim=True).clamp_min(1e-8)


def calibrate_scale_equivalence(
    model: nn.Module,
    calibration_data: Tensor,
    lr: float = 1e-3,
    steps: int = 100,
) -> dict[str, ScaleEquivalence]:
    """Calibrate OmniQuant scales for each TernairLinear layer.

    Only call this on a **trained** checkpoint, not on random init.
    """
    from ternair.quantization.linear import TernairLinear
    from ternair.quantization.ternary import ternary_linear_forward

    scales = {}
    for name, module in model.named_modules():
        if not isinstance(module, TernairLinear):
            continue
        scale_equi = ScaleEquivalence(module.in_features).to(calibration_data.device)
        optimizer  = torch.optim.AdamW(scale_equi.parameters(), lr=lr)
        x = calibration_data.clone().detach()
        w = module.weight.data.clone().detach()
        for _ in range(steps):
            y_ref       = F.linear(x, w)
            x_s, w_s   = scale_equi(x, w)
            gamma       = _compute_gamma(w_s, dim=-1)
            y_q         = F.linear(x_s, ternary_linear_forward(w_s, gamma))
            loss        = F.mse_loss(y_q, y_ref)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scales[name] = scale_equi
        print(f"  Calibrated {name}: scale={scale_equi.scale.mean().item():.4f}")
    return scales


__all__ = [
    "Activation8Bit",
    "apply_hadamard_transform",
    "apply_inverse_hadamard",
    "quantize_activations_8bit",
    "quantize_activations_8bit_forward",
    "ScaleEquivalence",
    "calibrate_scale_equivalence",
]
