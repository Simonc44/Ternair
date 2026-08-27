"""Ternary MLP block - SwiGLU activation for better training dynamics.

Implements a fully ternarised SwiGLU MLP:

    SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

All three projections (gate, up, down) use TernairLinear with
STE backward and optional learned alpha.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.model.norm import RMSNorm
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU / Swish).
    
    SiLU(x) = x * sigmoid(x)
    """
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.silu(x)


class TernairMLP(nn.Module):
    """SwiGLU MLP, all projections ternary.

    Layout::

        gate = ternary_linear(x)            # (B, T, I)  - gate
        up   = ternary_linear(x)            # (B, T, I)  - value
        h    = SiLU(gate) * up              # elementwise gating
        y    = ternary_linear(h)            # (B, T, H)  - output

    L'activation SwiGLU offre de meilleures performances que SquaredReLU
    sur les taches de language, notamment combinee a la quantification
    ternaire (BitNet b1.58 avec SwiGLU est la recommandation recente).
    """

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        S = config.hidden_size
        I = config.intermediate_size
        self.gate_proj = TernairLinear(S, I, bias=False, storage=config.storage)
        self.up_proj = TernairLinear(S, I, bias=False, storage=config.storage)
        self.down_proj = TernairLinear(I, S, bias=False, storage=config.storage)
        self.act = SiLU()

        # BitNet b1.58 sub-layer normalisation: applied on the gated
        # product before the down_proj projection.
        self._use_sub_norm: bool = bool(getattr(config, "use_sub_norm", False))
        if self._use_sub_norm:
            self.ffn_sub_norm = RMSNorm(I, eps=config.rms_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        # 8-bit activation quantisation before projection (BitNet paper)
        x_q = quantize_activations_8bit_forward(x)
        gate = self.gate_proj(x_q)
        up = self.up_proj(x_q)
        h = self.act(gate) * up
        if self._use_sub_norm:
            h = self.ffn_sub_norm(h)
        # Quantise before output projection
        h_q = quantize_activations_8bit_forward(h)
        return self.down_proj(h_q)


__all__ = ["TernairMLP", "SiLU"]
