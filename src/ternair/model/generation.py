"""Greedy generation loop (no KV cache - intentionally simple)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ternair.model.modeling import TernairForCausalLM


@torch.no_grad()
def generate(
    model: TernairForCausalLM,
    input_ids: Tensor,
    max_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
) -> Tensor:
    """Greedy decode ``max_new_tokens`` continuations of ``input_ids``.

    This is intentionally minimal - it re-runs the full forward pass
    on each generated step. A production version would add a KV cache.
    """
    out = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(out)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_token], dim=-1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return out


__all__ = ["generate"]
