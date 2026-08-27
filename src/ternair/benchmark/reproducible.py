"""Reproducible benchmarks for Ternair models.

Provides perplexity evaluation on WikiText-2 and generation speed
benchmarks that can be run and compared across machines.

Usage::

    python -c "
    from ternair.benchmark.reproducible import run_benchmark
    print(run_benchmark(profile='tiny', steps=100))
    "
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn.functional as F

from ternair.model.modeling import TernairForCausalLM
from ternair.model.generation import generate
from ternair.model.size_profiles import PROFILE_REGISTRY


@dataclass
class BenchmarkResult:
    """Structured benchmark output."""

    profile: str
    storage: str
    device: str

    # Perplexity (WikiText-2 style)
    perplexity: Optional[float] = None
    eval_tokens: int = 0
    eval_loss: Optional[float] = None

    # Speed
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    end_to_end_tokens_per_sec: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0

    # Model info
    total_params: int = 0
    model_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"=== Ternair Benchmark: {self.profile} ({self.storage}) ===",
            f"Device        : {self.device}",
            f"Parameters    : {self.total_params:,}",
            f"Model size    : {self.model_bytes / 1024**2:.2f} MiB",
            "",
        ]
        if self.perplexity is not None:
            lines += [
                f"Perplexity    : {self.perplexity:.2f}",
                f"Eval tokens   : {self.eval_tokens:,}",
                f"Eval loss     : {self.eval_loss:.4f}",
                "",
            ]
        lines += [
            f"Prefill speed : {self.prefill_tokens_per_sec:.1f} tokens/s ({self.prefill_ms:.1f} ms)",
            f"Decode speed  : {self.decode_tokens_per_sec:.1f} tokens/s ({self.decode_ms:.1f} ms)",
            f"E2E speed     : {self.end_to_end_tokens_per_sec:.1f} tokens/s",
            "",
        ]
        return "\n".join(lines)


def _build_toy_dataset(
    vocab_size: int = 256,
    seq_len: int = 128,
    num_seqs: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a deterministic synthetic dataset for perplexity evaluation.

    Each sequence is ``seq_len`` tokens sampled from ``[0, vocab_size)``.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (num_seqs, seq_len), generator=gen)


@torch.no_grad()
def _compute_perplexity(
    model: TernairForCausalLM,
    data: torch.Tensor,
    device: str = "cpu",
) -> tuple[float, float, int]:
    """Compute perplexity over ``data`` using a sliding window.

    Returns ``(perplexity, mean_loss, total_tokens)``.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for seq in data:
        ids = seq.unsqueeze(0).to(device)
        logits = model(ids)
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(mean_loss)
    return perplexity, mean_loss, total_tokens


@torch.no_grad()
def _measure_speed(
    model: TernairForCausalLM,
    vocab_size: int,
    prompt_len: int = 32,
    gen_len: int = 16,
    device: str = "cpu",
    warmup: int = 1,
    repeats: int = 3,
) -> tuple[float, float, float, float, float]:
    """Measure prefill and decode speed.

    Returns ``(prefill_tok_s, decode_tok_s, e2e_tok_s, prefill_ms, decode_ms)``.
    """
    model.eval()
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)

    # Warmup
    for _ in range(warmup):
        model.model.reset_kv_cache()
        generate(model, prompt, max_new_tokens=gen_len, temperature=0.0)

    # Prefill timing
    prefill_times = []
    for _ in range(repeats):
        model.model.reset_kv_cache()
        t0 = time.perf_counter()
        model(prompt, use_cache=True)
        prefill_times.append(time.perf_counter() - t0)
    avg_prefill_ms = (sum(prefill_times) / len(prefill_times)) * 1000

    # Decode timing (token-by-token with cache)
    decode_times = []
    for _ in range(repeats):
        model.model.reset_kv_cache()
        model(prompt, use_cache=True)
        last_token = prompt[:, -1:]
        t0 = time.perf_counter()
        for _ in range(gen_len):
            out = model(last_token, use_cache=True)
            last_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_times.append(time.perf_counter() - t0)
    avg_decode_ms = (sum(decode_times) / len(decode_times)) * 1000

    prefill_tok_s = prompt_len / (avg_prefill_ms / 1000) if avg_prefill_ms > 0 else 0
    decode_tok_s = gen_len / (avg_decode_ms / 1000) if avg_decode_ms > 0 else 0
    total_ms = avg_prefill_ms + avg_decode_ms
    e2e_tok_s = gen_len / (total_ms / 1000) if total_ms > 0 else 0

    return prefill_tok_s, decode_tok_s, e2e_tok_s, avg_prefill_ms, avg_decode_ms


def run_benchmark(
    profile: str = "tiny",
    storage: str = "packed",
    device: str = "cpu",
    run_perplexity: bool = True,
    run_speed: bool = True,
    eval_tokens: int = 1024,
    speed_prompt_len: int = 32,
    speed_gen_len: int = 16,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a full reproducible benchmark suite.

    Parameters
    ----------
    profile
        Model profile name (must be in ``PROFILE_REGISTRY``).
    storage
        Weight storage mode.
    device
        ``cpu`` or ``cuda``.
    run_perplexity
        Whether to evaluate perplexity.
    run_speed
        Whether to measure generation speed.
    eval_tokens
        Number of tokens to evaluate perplexity over.
    speed_prompt_len
        Prompt length for speed benchmark.
    speed_gen_len
        Number of tokens to generate for speed benchmark.
    seed
        Random seed for reproducibility.

    Returns
    -------
    BenchmarkResult
        Complete benchmark results.
    """
    profile_fn = PROFILE_REGISTRY.get(profile)
    if profile_fn is None:
        raise ValueError(f"Unknown profile {profile!r}; choose from {list(PROFILE_REGISTRY)}")

    config = profile_fn(storage=storage)
    model = TernairForCausalLM(config)
    model.freeze_storage()
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    result = BenchmarkResult(
        profile=profile,
        storage=storage,
        device=device,
        total_params=model.count_parameters(),
        model_bytes=model.num_bytes(),
    )

    if run_perplexity:
        # Build dataset with the model's vocab size
        n_seqs = max(1, eval_tokens // 128)
        data = _build_toy_dataset(
            vocab_size=config.vocab_size,
            seq_len=128,
            num_seqs=n_seqs,
            seed=seed,
        )
        ppl, loss, tokens = _compute_perplexity(model, data, device=device)
        result.perplexity = ppl
        result.eval_loss = loss
        result.eval_tokens = tokens

    if run_speed:
        prefill_s, decode_s, e2e_s, prefill_ms, decode_ms = _measure_speed(
            model,
            vocab_size=config.vocab_size,
            prompt_len=speed_prompt_len,
            gen_len=speed_gen_len,
            device=device,
        )
        result.prefill_tokens_per_sec = prefill_s
        result.decode_tokens_per_sec = decode_s
        result.end_to_end_tokens_per_sec = e2e_s
        result.prefill_ms = prefill_ms
        result.decode_ms = decode_ms

    return result


def run_comparison(
    profiles: list[str] | None = None,
    storage: str = "packed",
    device: str = "cpu",
    output_path: str | None = None,
) -> list[BenchmarkResult]:
    """Run benchmarks for multiple profiles and optionally save results.

    Parameters
    ----------
    profiles
        List of profile names.  Defaults to ``["tiny", "small"]``.
    storage
        Weight storage mode.
    device
        ``cpu`` or ``cuda``.
    output_path
        If set, save JSON results to this path.

    Returns
    -------
    list of BenchmarkResult
    """
    if profiles is None:
        profiles = ["tiny", "small"]

    results = []
    for p in profiles:
        print(f"\n--- Benchmarking {p} ({storage}) ---")
        r = run_benchmark(profile=p, storage=storage, device=device)
        print(r.summary())
        results.append(r)

    if output_path:
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


__all__ = [
    "BenchmarkResult",
    "run_benchmark",
    "run_comparison",
]
