"""Ternary MLP block - squared-ReLU activation to match BitNet b1.58."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear


class SquaredReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return torch.relu(x) ** 2


class TernairMLP(nn.Module):
    """Squared-ReLU MLP, all projections ternary.

    Layout::

        gate = ternary_linear(x)         # (B, T, I)
        up   = ternary_linear(x)         # (B, T, I)
        h    = squared_relu(gate) * up   # elementwise
        y    = ternary_linear(h)         # (B, T, H)
    """

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        S = config.hidden_size
        I = config.intermediate_size
        self.gate_proj = TernairLinear(S, I, bias=False, storage=config.storage)
        self.up_proj = TernairLinear(S, I, bias=False, storage=config.storage)
        self.down_proj = TernairLinear(I, S, bias=False, storage=config.storage)
        self.act = SquaredReLU()

    def forward(self, x: Tensor) -> Tensor:
        x_q = quantize_activations_8bit_forward(x)
        gate = self.gate_proj(x_q)
        up = self.up_proj(x_q)
        h = self.act(gate) * up
        h_q = quantize_activations_8bit_forward(h)
        return self.down_proj(h_q)


__all__ = ["TernairMLP", "SquaredReLU"]
