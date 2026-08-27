"""Top-level ternary causal LM.

This module stitches the blocks together, owns the embeddings / LM
head, and exposes :class:`TernairForCausalLM` for end-to-end demo /
training.

Training uses the in-line STE ternary weights (via
:class:`ternair.quantization.linear.TernairLinear`); once
:meth:`TernairForCausalLM.freeze_storage` is called the model switches
to the packed-trit buffer (or the int8 buffer) and uses ``γ`` only.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear
from ternair.model.block import RMSNorm, TernairBlock
from ternair.model.hybrid_block import TernairHybridBlock
from ternair.model.config import TernairConfig
from ternair.model.attention import _build_rope_cache


class TernairModel(nn.Module):
    """Backbone (no LM head)."""

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                TernairHybridBlock(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Cached RoPE buffers; lazily populated on first forward.
        self.register_buffer("_rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("_rope_sin", torch.empty(0), persistent=False)
        self._cached_seq_len = -1

    def reset_kv_cache(self) -> None:
        """Clear attention caches for a new generation request."""
        for layer in self.layers:
            if layer.is_attn:
                layer.block.reset_kv_cache()

    def _ensure_rope_cache(self, seq_len: int, device, dtype) -> tuple[Tensor, Tensor]:
        if seq_len <= self._cached_seq_len and self._rope_cos.numel() >= seq_len * self.config.head_dim:
            return (
                self._rope_cos[:seq_len].to(device=device, dtype=dtype),
                self._rope_sin[:seq_len].to(device=device, dtype=dtype),
            )
        cos, sin = _build_rope_cache(
            seq_len=seq_len,
            head_dim=self.config.head_dim,
            theta=self.config.rope_theta,
            device=device,
            dtype=dtype,
        )
        self._rope_cos = cos.detach()
        self._rope_sin = sin.detach()
        self._cached_seq_len = seq_len
        return cos, sin

    def forward(self, input_ids: Tensor, use_cache: bool = False) -> Tensor:  # type: ignore[override]
        x = self.embed_tokens(input_ids)
        x = quantize_activations_8bit_forward(x)
        seq_len = x.shape[1]
        # During decode with KV-cache, RoPE must cover the absolute
        # position of the new token (kv_cache_len).
        needed = seq_len
        if use_cache and not self.training:
            for layer in self.layers:
                if layer.is_attn:
                    attn_len = layer.block.attn._kv_cache_len
                    needed = max(needed, attn_len + seq_len)
                    break
        cos, sin = self._ensure_rope_cache(
            seq_len=needed, device=x.device, dtype=x.dtype
        )
        for block in self.layers:
            x = block(x, cos, sin, use_cache=use_cache)
        return self.norm(x)


class TernairForCausalLM(nn.Module):
    """Causal LM head (tied with the input embedding by default)."""

    def __init__(self, config: TernairConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TernairModel(config)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = TernairLinear(
                config.hidden_size, config.vocab_size, bias=False, storage=config.storage
            )

    def forward(self, input_ids: Tensor, use_cache: bool = False) -> Tensor:  # type: ignore[override]
        h = self.model(input_ids, use_cache=use_cache)
        if self.lm_head is None:
            logits = h @ self.model.embed_tokens.weight.T
        else:
            logits = self.lm_head(h)
        return logits

    # ------------------------------------------------------------------
    # Storage / export
    # ------------------------------------------------------------------
    @torch.no_grad()
    def freeze_storage(self) -> dict:
        """Switch all ternary linears to packed storage; return a snapshot."""
        from ternair.quantization.linear import TernairLinear

        snapshot: dict = {}
        for name, module in self.named_modules():
            if isinstance(module, TernairLinear):
                snapshot[name] = module.freeze_storage()
        return snapshot

    def count_parameters(self, include_embedding: bool = True) -> int:
        from ternair.quantization.linear import TernairLinear

        total = 0
        for _name, module in self.named_modules():
            if isinstance(module, TernairLinear):
                total += module.out_features * module.in_features
        if include_embedding:
            total += self.model.embed_tokens.weight.numel()
        return total

    def num_bytes(self, embedding_dtype_bytes: int = 2) -> int:
        """Estimate total on-disk footprint in bytes.

        * Each ternary linear contributes its packed-weight bytes + 4×γ.
        * Embeddings (and the LM head, if untied) count at the configured
          embedding dtype bytes per element.
        """
        from ternair.quantization.linear import TernairLinear

        total = 0
        for module in self.modules():
            if isinstance(module, TernairLinear):
                total += module.state_bytes()

        emb = self.model.embed_tokens.weight.numel() * embedding_dtype_bytes
        total += emb
        if self.lm_head is not None and not self.config.tie_word_embeddings:
            total += (
                self.lm_head.out_features
                * self.lm_head.in_features
                * embedding_dtype_bytes
            )
        return total


__all__ = ["TernairModel", "TernairForCausalLM"]
