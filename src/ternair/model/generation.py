"""Generation loop with temperature + top-k/top-p sampling."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ternair.model.modeling import TernairForCausalLM


def _top_k_top_p_filter(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = float("-inf"),
) -> Tensor:
    """Filter logits with top-k and/or nucleus (top-p) masking."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


@torch.no_grad()
def generate(
    model: TernairForCausalLM,
    input_ids: Tensor,
    max_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> Tensor:
    """Decode ``max_new_tokens`` continuations of ``input_ids``.

    Parameters
    ----------
    temperature : float
        Sampling temperature (1.0 = no scaling, <1.0 = sharper, >1.0 = flatter).
        If 0.0, falls back to greedy (argmax).
    top_k : int
        If >0, only sample from the top-k highest probability tokens.
    top_p : float
        If 0.0 < top_p < 1.0, use nucleus sampling (top-p cumulative mass).
    """
    out = input_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(out)
        logits = logits[:, -1, :]  # shape (1, vocab_size)

        if temperature == 0.0:
            # Greedy
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        out = torch.cat([out, next_token], dim=-1)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    return out


__all__ = ["generate"]
