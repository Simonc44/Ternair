"""Decoder-only attention block with ternary linear projections."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, T, D), cos/sin: (T, D) or (B, T, D)
    if cos.dim() < 3:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    else:
        cos = cos.unsqueeze(1)  # (B, 1, T, D)
    if sin.dim() < 3:
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class TernairAttention(nn.Module):
    """Attention block with ternary Q/K/V/O projections (GQA-capable)."""

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        self.config = config
        H = config.num_attention_heads
        KV = config.num_key_value_heads
        D = config.head_dim
        S = config.hidden_size

        self.q_proj = TernairLinear(S, H * D, bias=False, storage=config.storage)
        self.k_proj = TernairLinear(S, KV * D, bias=False, storage=config.storage)
        self.v_proj = TernairLinear(S, KV * D, bias=False, storage=config.storage)
        self.o_proj = TernairLinear(H * D, S, bias=False, storage=config.storage)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        B, T, _ = x.shape
        H = self.config.num_attention_heads
        KV = self.config.num_key_value_heads
        D = self.config.head_dim

        # 8-bit activation quantisation before projection (BitNet paper).
        q_in = quantize_activations_8bit_forward(x)
        k_in = quantize_activations_8bit_forward(x)
        v_in = quantize_activations_8bit_forward(x)

        q = self.q_proj(q_in).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(k_in).view(B, T, KV, D).transpose(1, 2)
        v = self.v_proj(v_in).view(B, T, KV, D).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Expand KV heads for grouped-query attention.
        if KV != H:
            repeat = H // KV
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H * D)

        # Quantise the attention output before the output projection.
        ctx_in = quantize_activations_8bit_forward(ctx)
        return self.o_proj(ctx_in)


__all__ = ["TernairAttention", "_build_rope_cache"]
