"""Ternary Selective State-Space block with parallel scan.

Fixes vs previous version
--------------------------
* Clamped log_A_cumsum before exp() to avoid overflow with random init.
  Without clamp, exp(large positive) = inf on the first forward pass
  when delta and A_log are random.
* Added output clamp to prevent NaN propagation downstream.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear

# Maximum absolute value for log_A_cumsum before exp()
# exp(20) ≈ 4.8e8, well within float32 range
_LOG_CLAMP = 20.0


def _selective_scan_parallel(
    x: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
) -> Tensor:
    """Vectorised selective scan (O(L) work, no Python loop over L).

    Parameters
    ----------
    x:      (B, L, D)  input
    delta:  (B, L, D)  softplus time step
    A:      (N,)       diagonal SSM matrix (must be negative)
    B:      (B, L, N)  input projection
    C:      (B, L, N)  output projection
    D:      (D,)       optional skip connection
    """
    B_size, L, D_size = x.shape
    N = A.shape[0]
    dtype = x.dtype
    device = x.device

    A = A.to(dtype=dtype, device=device)

    delta_4d = delta.unsqueeze(-1)                              # (B, L, D, 1)
    log_A_bar = delta_4d * A.view(1, 1, 1, N)                  # (B, L, D, N)

    # FIX: clamp log_A_cumsum before exp() to avoid overflow with random init.
    # A is supposed to be negative (stable system), but at init A_log can be
    # positive, making log_A_bar positive too. Without clamp:
    # exp(large positive) = inf → h = inf → y = NaN.
    log_A_bar = torch.clamp(log_A_bar, max=0.0)  # A must be negative for stability

    dBx = delta_4d * B.unsqueeze(2) * x.unsqueeze(-1)          # (B, L, D, N)

    log_A_cumsum = torch.cumsum(log_A_bar, dim=1)               # (B, L, D, N)
    log_A_cumsum = torch.clamp(log_A_cumsum, min=-_LOG_CLAMP, max=0.0)

    decay_term   = torch.exp(log_A_cumsum)                      # in (0, 1]
    anti_decay   = torch.exp(-log_A_cumsum)                     # in [1, inf)
    # clamp anti_decay to avoid blow-up when log_A_cumsum is near zero
    anti_decay   = torch.clamp(anti_decay, max=math.exp(_LOG_CLAMP))

    weighted_dBx    = anti_decay * dBx
    weighted_cumsum = torch.cumsum(weighted_dBx, dim=1)
    h = decay_term * weighted_cumsum                            # (B, L, D, N)

    y = (C.unsqueeze(2) * h).sum(dim=-1)                       # (B, L, D)

    if D is not None:
        y = y + D.to(dtype=dtype, device=device) * x

    # Final clamp to prevent NaN propagation if any residual instability
    y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
    return y


import math


class TernarySSMBlock(nn.Module):
    """SSM block with ternary projections and vectorised parallel scan."""

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        H = config.hidden_size
        self.hidden   = H
        self.ssm_dim  = getattr(config, "ssm_dim", 16)
        dt_rank_raw   = getattr(config, "ssm_dt_rank", "auto")
        self.dt_rank  = H // 4 if dt_rank_raw == "auto" else int(dt_rank_raw)
        self.expand   = 2

        self.x_proj      = TernairLinear(H, H * self.expand, bias=False, storage=config.storage)
        self.dt_rank_proj = TernairLinear(H, self.dt_rank,   bias=False, storage=config.storage)
        self.dt_proj     = nn.Linear(self.dt_rank, H)
        self.B_proj      = TernairLinear(H, self.ssm_dim, bias=False, storage=config.storage)
        self.C_proj      = TernairLinear(H, self.ssm_dim, bias=False, storage=config.storage)

        # A_log initialised to log(1..N) — ensures A = -exp(A_log) < 0
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.ssm_dim + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(H))

        self.out_proj = TernairLinear(H, H, bias=False, storage=config.storage)
        self.norm     = nn.LayerNorm(H)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x_q  = quantize_activations_8bit_forward(x)
        xz   = self.x_proj(x_q)                                  # (B, L, 2H)
        x_gate, z = xz.chunk(2, dim=-1)                          # (B, L, H) each

        dt_r  = self.dt_rank_proj(quantize_activations_8bit_forward(x_gate))
        delta  = F.softplus(self.dt_proj(dt_r))                  # (B, L, H) > 0
        B_s    = self.B_proj(quantize_activations_8bit_forward(x_gate))  # (B, L, N)
        C_s    = self.C_proj(quantize_activations_8bit_forward(x_gate))  # (B, L, N)

        # A = -exp(A_log) is always negative → stable system
        A = -torch.exp(self.A_log.to(dtype=x.dtype, device=x.device))   # (N,)

        y   = _selective_scan_parallel(x_gate, delta, A, B_s, C_s, D=self.D)
        y   = y * torch.sigmoid(z)
        out = self.out_proj(quantize_activations_8bit_forward(y))
        return self.norm(residual + out)
