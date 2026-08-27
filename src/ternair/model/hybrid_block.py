"""Hybrid block — attention *or* SSM per layer.

The :class:`TernairHybridBlock` dispatches to either:

* :class:`TernairBlock`  (GQA + MLP) for layers with ``layer_idx < num_attn_layers``.
* :class:`TernarySSMBlock` for the remaining layers.

This lets the user configure a mixed model where early layers use
full attention (learning rich representations) and later layers use
the O(1) SSM recurrence (long-context processing without KV cache).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ternair.model.attention import TernairAttention, _build_rope_cache
from ternair.model.config import TernairConfig
from ternair.model.mlp import TernairMLP
from ternair.model.block import RMSNorm, TernairBlock
from ternair.model.ssm import TernarySSMBlock


class TernairHybridBlock(nn.Module):
    """Either an attention block or an SSM block.

    Parameters
    ----------
    config
        The model config.  The layer is an attention block iff
        ``layer_idx < config.num_attn_layers``.  Otherwise it uses
        the SSM block.
    layer_idx
        The layer index in the transformer stack.

    Notes
    -----
    La distribution des couches suit un pattern periodique defini par
    ``attn_layer_period`` (par defaut: 4 = 3x SSM + 1x Attention).
    
    Pattern 3:1 (SSM x 3, Attention x 1) :
      - Layer 0 : SSM
      - Layer 1 : SSM
      - Layer 2 : SSM
      - Layer 3 : Attention
      - Layer 4 : SSM
      - ...
    """

    def __init__(self, config: TernairConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        
        attn_layers = getattr(config, "num_attn_layers", config.num_hidden_layers)
        period = getattr(config, "attn_layer_period", 0)

        if period > 1 and attn_layers == config.num_hidden_layers:
            self.is_attn = ((layer_idx + 1) % period == 0)
        else:
            self.is_attn = layer_idx < attn_layers

        if self.is_attn:
            self.block = TernairBlock(config, layer_idx)
        else:
            self.ssm_block = TernarySSMBlock(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # type: ignore[override]
        if self.is_attn:
            return self.block(x, cos, sin)
        # SSM blocks ignore RoPE; pass through unchanged
        return self.ssm_block(x)


__all__ = ["TernairHybridBlock"]
