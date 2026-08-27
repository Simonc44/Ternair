"""Generation loop with advanced sampling, streaming, and chat templates.

Features
--------
* **Greedy decode** (temperature = 0.0)
* **Temperature sampling** with top-K and top-P (nucleus) filtering
* **Repetition penalty** to discourage repeating tokens
* **Streaming generator** (``generate_stream``) with ``yield`` per token
* **Chat templates** for ChatML and Llama-3 formats
"""

from __future__ import annotations

from typing import Generator, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ternair.model.modeling import TernairForCausalLM


# ---------------------------------------------------------------------------
# Logit processors
# ---------------------------------------------------------------------------

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


def _apply_repetition_penalty(
    logits: Tensor,
    token_ids: Tensor,
    penalty: float = 1.0,
) -> Tensor:
    """Apply repetition penalty to discourage repeating tokens.

    For each token that has already appeared in ``token_ids``, the
    logit is divided by ``penalty`` (if penalty > 1) or multiplied
    by ``penalty`` (if penalty < 1).

    Args:
        logits: (1, vocab_size) raw logits
        token_ids: (1, T) sequence of already-generated token IDs
        penalty: Repetition penalty (> 1.0 = discourage repeats)
    """
    if penalty == 1.0:
        return logits

    # Get unique tokens that have appeared
    unique_tokens = torch.unique(token_ids)
    if len(unique_tokens) == 0:
        return logits

    if penalty > 1.0:
        # Divide logits for repeated tokens (discourage)
        logits[:, unique_tokens] /= penalty
    else:
        # Multiply logits for repeated tokens (encourage, rare)
        logits[:, unique_tokens] *= penalty

    return logits


def _sample_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> Tensor:
    """Sample a single token from logits with temperature and filtering."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Standard generation (returns full sequence)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: TernairForCausalLM,
    input_ids: Tensor,
    max_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    pad_token_id: Optional[int] = None,
) -> Tensor:
    """Decode ``max_new_tokens`` continuations of ``input_ids``.

    Parameters
    ----------
    model:
        The Ternair model (eval mode recommended).
    input_ids:
        (1, T) prompt tokens.
    max_new_tokens:
        Maximum number of tokens to generate.
    eos_token_id:
        If not None, generation stops when this token is produced.
    temperature:
        Sampling temperature. 0.0 = greedy, 1.0 = unchanged.
    top_k:
        If > 0, only sample from the top-K highest probability tokens.
    top_p:
        If 0.0 < top_p < 1.0, use nucleus sampling.
    repetition_penalty:
        > 1.0 to penalise already-seen tokens (default 1.0 = no penalty).
    pad_token_id:
        Padding token ID (used for EOS if not provided).

    Returns
    -------
    (1, T + generated) tokens including the prompt.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    out = input_ids.clone()
    eos = eos_token_id if eos_token_id is not None else pad_token_id

    for step in range(max_new_tokens):
        logits = model(out)
        logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, out, penalty=repetition_penalty)

        next_token = _sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        out = torch.cat([out, next_token], dim=-1)

        if eos is not None and int(next_token.item()) == eos:
            break

    return out


# ---------------------------------------------------------------------------
# Streaming generation (yields tokens one at a time)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_stream(
    model: TernairForCausalLM,
    input_ids: Tensor,
    max_new_tokens: int = 64,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    pad_token_id: Optional[int] = None,
) -> Generator[Tensor, None, list[int]]:
    """Streaming generator that yields each new token.

    Usage::

        for token_tensor in generate_stream(model, prompt, max_new_tokens=32):
            token_id = token_tensor.item()
            # Send to UI, console, etc.

    After iteration, the full token list is returned via ``.value``
    (PEP 342 / 479).

    Yields
    ------
    Tensor
        Scalar tensor with the latest generated token ID.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    out = input_ids.clone()
    eos = eos_token_id if eos_token_id is not None else pad_token_id
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = model(out)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, out, penalty=repetition_penalty)

        next_token = _sample_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        out = torch.cat([out, next_token], dim=-1)
        token_id = int(next_token.item())
        generated.append(token_id)

        yield next_token.squeeze(0)  # (1,) → scalar view

        if eos is not None and token_id == eos:
            break

    return generated


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

ChatFormat = Literal["chatml", "llama3", "raw"]


def format_chat_prompt(
    messages: list[dict[str, str]],
    tokenizer: Optional["PreTrainedTokenizer"] = None,  # type: ignore[name-defined]  # noqa: F821
    format: ChatFormat = "chatml",  # noqa: A002
    add_generation_prompt: bool = True,
) -> str:
    """Format a list of chat messages into a single prompt string.

    Supports three formats:

    * ``"chatml"`` — ``<|im_start|>role\\nmessage<|im_end|>``
    * ``"llama3"`` — ``<|start_header_id|>role<|end_header_id|>\\n\\nmessage<|eot_id|>``
    * ``"raw"`` — ``role: message\\n`` (simple, for debugging)

    Parameters
    ----------
    messages:
        List of ``{"role": ..., "content": ...}`` dicts.
    tokenizer:
        Optional tokenizer (if provided, uses its chat_template).
    format:
        Which template to use when ``tokenizer`` is None.
    add_generation_prompt:
        Whether to append the assistant header (so the model continues).

    Returns
    -------
    str
        The formatted prompt string.
    """
    # If a tokenizer with a chat_template is provided, use it
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass  # Fall through to built-in templates

    # Built-in templates
    if format == "chatml":
        pieces = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            pieces.append("<|im_start|>assistant\n")
        return "\n".join(pieces)

    elif format == "llama3":
        pieces = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            pieces.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        if add_generation_prompt:
            pieces.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(pieces)

    else:  # raw
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        ) + ("\nassistant:" if add_generation_prompt else "")


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def decode_tokens(
    token_ids: Sequence[int],
    tokenizer,
    skip_special_tokens: bool = True,
) -> str:
    """Decode a sequence of token IDs to text.

    Works with any tokenizer that has a ``decode`` method (HuggingFace,
    tiktoken, etc.).
    """
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    return " ".join(str(t) for t in token_ids)


__all__ = [
    "generate",
    "generate_stream",
    "format_chat_prompt",
    "decode_tokens",
]
