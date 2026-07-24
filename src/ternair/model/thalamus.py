"""Thalamic bottleneck — K-WTA compression module.

Inspired by the thalamus in neurobiology, this module compresses
variable-length sensory inputs (image patches, audio frames) into a
fixed set of K latent tokens before feeding them to the transformer.

The design combines:

1. **K learned latent queries** — ``nn.Embedding(K, hidden)`` that are
   shared across all inputs (similar to Perceiver's latent array).
2. **Cross-attention** — each latent query attends over all input
   tokens. The key/value projections are ternary (BitNet-style).
3. **K-WTA** — the attention logits are sparsified: only the top-K
   source positions contribute per query (``topk K`` on the attention
   matrix). This mirrors the competitive selection hypothesis of the
   fruit-fly mushroom body.

The output is a fixed ``(K, hidden)`` sequence of compressed latent
tokens, regardless of the input sequence length.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear


class ThalamicBottleneck(nn.Module):
    """Compress ``(B, N, D_in)`` → ``(B, K, D)`` via cross-attn + K-WTA.

    Parameters
    ----------
    config
        A :class:`TernairConfig` instance.  New fields used::

            thalamus_k       — number of latent tokens (default 32)
            thalamus_heads   — number of cross-attn heads (default 4)
    input_dim
        Feature dimension of the incoming tokens.  If ``input_dim
        != config.hidden_size``, an initial linear projection is applied.
    """

    def __init__(self, config: TernairConfig, input_dim: int | None = None) -> None:
        super().__init__()
        self.config = config

        K = getattr(config, "thalamus_k", 32)
        heads = getattr(config, "thalamus_heads", 4)
        # Safe lookup — ensure no -1 leaks through (e.g. if __post_init__ hasn't run)
        raw_embed = getattr(config, "thalamus_dim", config.hidden_size)
        if raw_embed is None or raw_embed <= 0:
            raw_embed = config.hidden_size
        embed_dim = raw_embed
        input_dim = input_dim or config.hidden_size
        self.K = K
        self.heads = heads
        self.embed_dim = embed_dim

        if input_dim != embed_dim:
            self.input_proj = TernairLinear(input_dim, embed_dim, bias=False, storage=config.storage)
        else:
            self.input_proj = nn.Identity()

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(K, embed_dim))

        # Cross-attention projections (ternary)
        self.q_proj = TernairLinear(embed_dim, embed_dim, bias=False, storage=config.storage)
        self.k_proj = TernairLinear(embed_dim, embed_dim, bias=False, storage=config.storage)
        self.v_proj = TernairLinear(embed_dim, embed_dim, bias=False, storage=config.storage)
        self.o_proj = TernairLinear(embed_dim, embed_dim, bias=False, storage=config.storage)

        # Post-attention MLP (ternary)
        self.mlp_gate = TernairLinear(embed_dim, embed_dim, bias=False, storage=config.storage)
        self.mlp_down = TernairLinear(embed_dim, embed_dim, bias=False, storage=config.storage)

        self.norm_cross = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Compress ``x`` to ``K`` latent tokens.

        Parameters
        ----------
        x : (B, N, D_in)
            Variable-length input tokens.

        Returns
        -------
        latents : (B, K, embed_dim)
            Compressed latent representation.
        """
        B, N, _ = x.shape

        # Project input if needed + 8-bit activation quant
        x = self.input_proj(x)
        x = quantize_activations_8bit_forward(x)

        # Broadcast latent queries to batch
        q = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        q = quantize_activations_8bit_forward(q)

        # Cross-attention (K = queries, N = key/val)
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)

        # Multi-head
        H = self.heads
        D = self.embed_dim // H
        q_mh = q_proj.view(B, self.K, H, D).transpose(1, 2)  # (B, H, K, D)
        k_mh = k_proj.view(B, N, H, D).transpose(1, 2)      # (B, H, N, D)
        v_mh = v_proj.view(B, N, H, D).transpose(1, 2)

        # K-WTA on the source dimension: keep only top-K source positions        
        # Handle case N < K by reducing K_WTA
        k_wta = min(self.K, N)
        scores = torch.matmul(q_mh, k_mh.transpose(-2, -1)) / math.sqrt(D)  # (B, H, K, N)

        # Simpler approach: mask out non-top-K positions in the softmax
        topk_vals, topk_idx = scores.topk(k_wta, dim=-1)  # (B, H, K, k_wta)
        mask = torch.full((B, H, self.K, N), float("-inf"), device=x.device, dtype=x.dtype)
        mask.scatter_(-1, topk_idx, 0.0)  # top-K positions get 0 (neutral logit)
        scores_masked = scores + mask
        attn = torch.softmax(scores_masked, dim=-1)  # (B, H, K, N)

        ctx = torch.matmul(attn, v_mh)  # (B, H, K, D)
        ctx = ctx.transpose(1, 2).contiguous().view(B, self.K, self.embed_dim)

        out = self.norm_cross(q + self.o_proj(quantize_activations_8bit_forward(ctx)))

        # MLP
        gate = self.mlp_gate(quantize_activations_8bit_forward(out))
        h = torch.relu(gate) ** 2
        out = out + self.mlp_down(quantize_activations_8bit_forward(h))
        out = self.norm_mlp(out)

        return out
