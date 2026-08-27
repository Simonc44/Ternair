"""Ternary decoder block."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ternair.model.attention import TernairAttention, _build_rope_cache
from ternair.model.config import TernairConfig
from ternair.model.mlp import TernairMLP
from ternair.model.norm import RMSNorm  # re-exported below for back-compat


class TernairBlock(nn.Module):
    def __init__(self, config: TernairConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = TernairAttention(config)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TernairMLP(config)
        self.layer_idx = layer_idx

    def reset_kv_cache(self) -> None:
        """Clear this block's attention cache."""
        self.attn.reset_kv_cache()

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, use_cache: bool = False) -> Tensor:  # type: ignore[override]
        x_norm = self.ln_1(x)
        x = x + self.attn(x_norm, cos, sin, use_cache=use_cache)
        x_norm = self.ln_2(x)
        x = x + self.mlp(x_norm)
        return x


__all__ = ["TernairBlock", "RMSNorm"]
_ = _build_rope_cache  # re-export convenience
