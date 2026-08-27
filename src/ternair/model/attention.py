"""Decoder-only attention block with ternary linear projections."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    if cos.dim() < 3:
        cos = cos.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(1)
    if sin.dim() < 3:
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def _build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype
) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class TernaryKVQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, bits: int = 2) -> Tensor:  # type: ignore[override]
        block_size = 32
        B, H, T, D = x.shape
        # Pad T to multiple of block_size
        pad = (block_size - T % block_size) % block_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        T_pad = x.shape[2]
        x_blocks = x.view(B, H, T_pad // block_size, block_size, D)
        absmax = x_blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        scale = absmax / (2 ** (bits - 1) - 1)
        q = torch.clamp(
            torch.round(x_blocks / scale),
            -(2 ** (bits - 1)),
            2 ** (bits - 1) - 1,
        )
        out = (q * scale).view(B, H, T_pad, D)
        return out[:, :, :T - pad if pad else T, :]

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        return grad_out, None


def _quantize_kv(x: Tensor, bits: int = 2) -> Tensor:
    return TernaryKVQuant.apply(x, bits)


class TernairAttention(nn.Module):
    """GQA attention with ternary projections and causal mask.

    Fixes vs previous version
    --------------------------
    * Added causal mask in training mode — missing mask caused
      -inf scores → NaN in softmax.
    * Single quantize_activations_8bit_forward call on x before
      all three projections (was called 3× separately).
    * Used F.scaled_dot_product_attention with is_causal=True
      which handles the mask and numerics more robustly.
    """

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        self.config = config
        H  = config.num_attention_heads
        KV = config.num_key_value_heads
        D  = config.head_dim
        S  = config.hidden_size

        self.q_proj = TernairLinear(S, H * D,  bias=False, storage=config.storage)
        self.k_proj = TernairLinear(S, KV * D, bias=False, storage=config.storage)
        self.v_proj = TernairLinear(S, KV * D, bias=False, storage=config.storage)
        self.o_proj = TernairLinear(H * D, S,  bias=False, storage=config.storage)

        self._kv_cache_k: Tensor | None = None
        self._kv_cache_v: Tensor | None = None
        self._kv_cache_len: int = 0
        self._use_kv_quant: bool = getattr(config, "kv_cache_bits", 0) > 0
        self._kv_bits: int = getattr(config, "kv_cache_bits", 2)

    def _reset_kv_cache(self) -> None:
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_len = 0

    def reset_kv_cache(self) -> None:
        """Clear cached keys and values before a new independent sequence."""
        self._reset_kv_cache()

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        use_cache: bool = False,
    ) -> Tensor:
        B, T, _ = x.shape
        H  = self.config.num_attention_heads
        KV = self.config.num_key_value_heads
        D  = self.config.head_dim

        # Single quantisation call for all three projections
        x_q = quantize_activations_8bit_forward(x)

        q = self.q_proj(x_q).view(B, T, H,  D).transpose(1, 2)   # (B, H, T, D)
        k = self.k_proj(x_q).view(B, T, KV, D).transpose(1, 2)   # (B, KV, T, D)
        v = self.v_proj(x_q).view(B, T, KV, D).transpose(1, 2)   # (B, KV, T, D)

        # Determine if this is a decode step (single token after prefill).
        is_decode = use_cache and not self.training and self._kv_cache_k is not None

        if is_decode:
            # Decode: apply RoPE at absolute position for the new token.
            pos = self._kv_cache_len
            q = _apply_rope(q, cos[pos:pos+1], sin[pos:pos+1])
            k = _apply_rope(k, cos[pos:pos+1], sin[pos:pos+1])
            # Append to cache
            self._kv_cache_k = torch.cat([self._kv_cache_k, k], dim=2)
            self._kv_cache_v = torch.cat([self._kv_cache_v, v], dim=2)
            k_full = self._kv_cache_k
            v_full = self._kv_cache_v
            self._kv_cache_len = k_full.shape[2]
        elif use_cache and not self.training:
            # Prefill: apply RoPE for all positions, then cache.
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            if self._use_kv_quant:
                self._kv_cache_k = _quantize_kv(k, bits=self._kv_bits)
                self._kv_cache_v = _quantize_kv(v, bits=self._kv_bits)
            else:
                self._kv_cache_k = k
                self._kv_cache_v = v
            self._kv_cache_len = self._kv_cache_k.shape[2]
            k_full = self._kv_cache_k
            v_full = self._kv_cache_v
        else:
            # Training / no cache: standard full-sequence path.
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
            k_full = k
            v_full = v

        # GQA head expansion
        if KV != H:
            repeat = H // KV
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        ctx = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=not is_decode,
        )  # (B, H, T, D)

        ctx = ctx.transpose(1, 2).contiguous().view(B, T, H * D)
        ctx_q = quantize_activations_8bit_forward(ctx)
        return self.o_proj(ctx_q)


__all__ = ["TernairAttention", "_build_rope_cache"]
