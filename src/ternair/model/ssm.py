"""Ternary Selective State-Space block with parallel scan.

Numerical design
----------------
* Per-step log-decay is clamped to [-_LOG_CLAMP, 0] before exp(): a
  positive A at init (or a large negative delta*A) would otherwise make
  exp() overflow.  Clamping each *step* -- not the cumulative sum --
  preserves the differences S_t - S_i that the scan needs.
* The sequence is processed in fixed-size chunks: inside a chunk the
  anti-decay exp(-S) stays <= exp(4 * 20) = exp(80) < float32 max, so
  the decay/anti-decay decomposition is exact.  Across chunks a short
  sequential carry propagates the state; every value there is bounded
  (decay <= 1, state ~ O(|dBx|)), so it never overflows either.
* Output clamp prevents NaN propagation downstream (safety net only).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear

# Maximum absolute value of a single log-decay step before exp().
# exp(20) ~= 4.8e8, well within float32 range.
_LOG_CLAMP = 20.0

# Chunk size for the parallel scan.  Within a chunk the anti-decay
# exp(-S) is bounded by exp(BC * _LOG_CLAMP); BC * _LOG_CLAMP must stay
# below ~88 (the fp32 exp() overflow threshold), so BC = 4 is safe.
_SCAN_CHUNK = 4


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
    log_A_bar = torch.clamp(
        delta_4d * A.view(1, 1, 1, N),
        min=-_LOG_CLAMP,
        max=0.0,
    )                                                        # (B, L, D, N)

    dBx = delta_4d * B.unsqueeze(2) * x.unsqueeze(-1)          # (B, L, D, N)

    # ------------------------------------------------------------------
    # Chunked scan: within-chunk parallel decomposition (exact) + a short
    # sequential carry across chunks (bounded).  See module docstring.
    # ------------------------------------------------------------------
    BC = _SCAN_CHUNK
    L_orig = L
    if L % BC:
        # F.pad pads pairs from the LAST dimension backwards, so six
        # entries = (N: 0, 0), (D: 0, 0), (L: 0, pad).
        pad = BC - (L % BC)
        log_A_bar = F.pad(log_A_bar, (0, 0, 0, 0, 0, pad))
        dBx = F.pad(dBx, (0, 0, 0, 0, 0, pad))
        L = L + pad
    n_chunks = L // BC
    lb = log_A_bar.view(B_size, n_chunks, BC, D_size, N)       # (B, C, BC, D, N)
    dbx = dBx.view(B_size, n_chunks, BC, D_size, N)

    S_local = torch.cumsum(lb, dim=2)                          # (B, C, BC, D, N)
    decay_local = torch.exp(S_local)                           # in (0, 1]
    anti_local = torch.exp(-S_local)                           # <= exp(80), exact
    G_local = torch.cumsum(anti_local * dbx, dim=2)
    h_local = decay_local * G_local                            # (B, C, BC, D, N)

    h_end = h_local[:, :, -1]                                  # (B, C, D, N)
    decay_end = decay_local[:, :, -1]                          # (B, C, D, N)

    # Sequential carry: state_c is the hidden state at the end of chunk
    # c-1.  Only O(n_chunks) tiny (B, D, N) ops -- negligible next to the
    # vectorised within-chunk work.
    state = torch.zeros(B_size, D_size, N, dtype=dtype, device=device)
    states = [state]
    for c in range(n_chunks):
        state = decay_end[:, c] * state + h_end[:, c]
        states.append(state)
    state_stack = torch.stack(states[:-1], dim=1)              # (B, C, D, N)

    h = decay_local * state_stack.unsqueeze(2) + h_local       # (B, C, BC, D, N)
    h = h.reshape(B_size, n_chunks * BC, D_size, N)[:, :L_orig]

    y = (C.unsqueeze(2) * h).sum(dim=-1)                       # (B, L, D)

    if D is not None:
        y = y + D.to(dtype=dtype, device=device) * x

    # Final clamp to prevent NaN propagation if any residual instability
    y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
    return y


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

        # NOTE: quantise x_gate once and reuse it for the three projections
        # (dt_rank / B / C).  Quantising the same tensor three times is
        # redundant work and measurably slows down prefill.
        x_gate_q = quantize_activations_8bit_forward(x_gate)
        dt_r  = self.dt_rank_proj(x_gate_q)
        delta  = F.softplus(self.dt_proj(dt_r))                  # (B, L, H) > 0
        B_s    = self.B_proj(x_gate_q)                            # (B, L, N)
        C_s    = self.C_proj(x_gate_q)                            # (B, L, N)

        # A = -exp(A_log) is always negative → stable system
        A = -torch.exp(self.A_log.to(dtype=x.dtype, device=x.device))   # (N,)

        y   = _selective_scan_parallel(x_gate, delta, A, B_s, C_s, D=self.D)
        y   = y * torch.sigmoid(z)
        out = self.out_proj(quantize_activations_8bit_forward(y))
        return self.norm(residual + out)
