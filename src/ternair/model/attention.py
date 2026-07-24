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


class TernaryKVQuant(torch.autograd.Function):
    """Quantise K/V cache to 2-bit with per-block scaling."""

    @staticmethod
    def forward(ctx, x: Tensor, bits: int = 2) -> Tensor:  # type: ignore[override]
        # Per-block scale (block_size = 32 tokens)
        block_size = 32
        B, H, T, D = x.shape

        # Reshape to group by blocks
        x_blocks = x.view(B, H, -1, block_size, D)
        absmax = x_blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        scale = absmax / (2 ** (bits - 1) - 1)
        q = torch.clamp(torch.round(x_blocks / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        # STE: gradient passe a travers
        out = q * scale
        return out.view(B, H, T, D)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        return grad_out, None


def _quantize_kv(x: Tensor, bits: int = 2) -> Tensor:
    """Quantize K/V tensor with 2-bit precision using block scaling."""
    return TernaryKVQuant.apply(x, bits)


class TernairAttention(nn.Module):
    """Attention block with ternary Q/K/V/O projections (GQA-capable).

    Nouveau : KV-Cache quantifie en 2-bit (BitAttention) pour reduire
    l'empreinte memoire du contexte long.
    """

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

        # KV-Cache quantifie
        self._kv_cache_k: Tensor | None = None
        self._kv_cache_v: Tensor | None = None
        self._kv_cache_len: int = 0
        self._use_kv_quant: bool = getattr(config, "kv_cache_bits", 0) > 0
        self._kv_bits: int = getattr(config, "kv_cache_bits", 2)

    def _reset_kv_cache(self) -> None:
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_len = 0

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

        q_in = quantize_activations_8bit_forward(x)
        k_in = quantize_activations_8bit_forward(x)
        v_in = quantize_activations_8bit_forward(x)

        q = self.q_proj(q_in).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(k_in).view(B, T, KV, D).transpose(1, 2)
        v = self.v_proj(v_in).view(B, T, KV, D).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # KV-Cache quantifie (BitAttention)
        if not self.training and self._use_kv_quant:
            # Quantifier K/V avant de les ajouter au cache
            k_q = _quantize_kv(k, bits=self._kv_bits)
            v_q = _quantize_kv(v, bits=self._kv_bits)

            if self._kv_cache_k is None:
                self._kv_cache_k = k_q
                self._kv_cache_v = v_q
            else:
                self._kv_cache_k = torch.cat([self._kv_cache_k, k_q], dim=2)
                self._kv_cache_v = torch.cat([self._kv_cache_v, v_q], dim=2)

            k = self._kv_cache_k
            v = self._kv_cache_v
            self._kv_cache_len = k.shape[2]

        # Expand KV heads for grouped-query attention.
        if KV != H:
            repeat = H // KV
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H * D)

        ctx_in = quantize_activations_8bit_forward(ctx)
        return self.o_proj(ctx_in)


__all__ = ["TernairAttention", "_build_rope_cache"]
