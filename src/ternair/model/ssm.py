"""Ternary Selective State-Space block — Mamba-style.

This module implements a selective SSM where the state-transition
matrices A, B, C are **input-dependent** (via ternary projections),
and the recurrence is evaluated with a sequential scan (O(N) time,
O(1) memory for inference).  This completely eliminates the KV cache
during long-sequence generation.

Key differences from regular Mamba
-----------------------------------
* The linear projections (x_proj, B_proj, C_proj, dt_proj, out_proj)
  use :class:`TernairLinear` with the fastpacked/packed storage.
* The SSM state size ``ssm_dim`` is kept small (default 16) so the
  recurrent state is negligible.
* The sequential scan is used instead of the parallel associative
  scan for simplicity — during training the call is fully
  differentiable via autograd.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ternair.model.config import TernairConfig
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear


class TernarySSMBlock(nn.Module):
    """SSM block with ternary projections.

    Parameters
    ----------
    config
        A :class:`TernairConfig` instance.  New fields used::

            ssm_dim   — SSM state size per channel (default 16)
            ssm_dt_rank — Δ projection rank (default ``auto`` = hidden_size // 4)
    """

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        H = config.hidden_size
        self.hidden = H
        self.ssm_dim: int = getattr(config, "ssm_dim", 16)
        dt_rank_raw = getattr(config, "ssm_dt_rank", "auto")
        self.dt_rank: int = H // 4 if dt_rank_raw == "auto" else int(dt_rank_raw)
        self.expand = 2  # expand factor

        # Project input → (z, x) for the gated structure
        self.x_proj = TernairLinear(H, H * self.expand, bias=False, storage=config.storage)
        # Δ, B, C projections (input-dependent) — operate on x (dim H), not the expanded dim
        self.dt_proj = nn.Linear(self.dt_rank, H)  # kept FP; very small
        self.dt_rank_proj = TernairLinear(H, self.dt_rank, bias=False, storage=config.storage)
        self.B_proj = TernairLinear(H, self.ssm_dim, bias=False, storage=config.storage)
        self.C_proj = TernairLinear(H, self.ssm_dim, bias=False, storage=config.storage)

        # Learned SSM parameters
        # A_log → A = -exp(A_log) ∈ ℝ^{ssm_dim}, diagonal, negative, learned
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.ssm_dim + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(H))

        # Output projection (y already has dim H after gating)
        self.out_proj = TernairLinear(H, H, bias=False, storage=config.storage)
        self.norm = nn.LayerNorm(H)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Process ``x`` of shape ``(B, L, H)`` → ``(B, L, H)``."""
        B, L, H = x.shape
        residual = x

        # 1) Input projection + gating
        # 8-bit activation quant before ternary matmul
        x_in = quantize_activations_8bit_forward(x)
        xz = self.x_proj(x_in)  # (B, L, 2H)
        x, z = xz.chunk(2, dim=-1)  # both (B, L, H)

        # 2) Selective SSM parameters
        # Δ = softplus(linear_dt(linear_dt_rank(x)))
        dt_r = self.dt_rank_proj(quantize_activations_8bit_forward(x))  # (B, L, dt_rank)
        delta = F.softplus(self.dt_proj(dt_r))  # (B, L, 2H)  — expanded

        B_s = self.B_proj(quantize_activations_8bit_forward(x))  # (B, L, ssm_dim)
        C_s = self.C_proj(quantize_activations_8bit_forward(x))  # (B, L, ssm_dim)

        A = -torch.exp(self.A_log).to(dtype=x.dtype, device=x.device)  # (ssm_dim,)

        # 3) Selective scan (sequential — O(L) steps, O(1) memory)
        y = self._selective_scan(x, delta, A, B_s, C_s, D_param=self.D)  # (B, L, H)

        # 4) Gating + output
        y = y * torch.sigmoid(z)
        out = self.out_proj(quantize_activations_8bit_forward(y))
        return self.norm(residual + out)

    # ------------------------------------------------------------------
    # Selective scan
    # ------------------------------------------------------------------
    @staticmethod
    def _selective_scan(
        x: Tensor,     # (B, L, D)
        delta: Tensor, # (B, L, D)  (D = H)
        A: Tensor,     # (N,)  N = ssm_dim
        B: Tensor,     # (B, L, N)
        C: Tensor,     # (B, L, N)
        D_param: Tensor | None = None,  # (D,) optional skip connection
    ) -> Tensor:
        """Sequential scan over the L dimension.

        Uses a loop over time for clarity.  A production version should
        replace this with the parallel associative scan from the Mamba
        paper.

        State equation::
            h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
            y_t = C_t * h_t + D * x_t
        """
        B_size, L, D = x.shape
        N = A.shape[0]
        device = x.device
        dtype = x.dtype

        h = torch.zeros(B_size, D, N, device=device, dtype=dtype)
        ys: list[Tensor] = []

        for t in range(L):
            dt = delta[:, t, :].unsqueeze(-1)  # (B, D, 1)
            A_bar = torch.exp(dt * A)           # (B, D, N)
            B_bar = dt * B[:, t, :].unsqueeze(1)  # (B, 1, N) → broadcast to (B, D, N)
            x_t = x[:, t, :]                    # (B, D)
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # (B, D, N)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, D)
            if D_param is not None:
                y_t = y_t + D_param * x_t
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, D)
