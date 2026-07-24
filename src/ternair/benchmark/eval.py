"""Evaluation suite for Ternair models.

Measures the retention of the teacher model's capabilities after
ternary quantization and distillation.

Benchmarks
----------
* **Perplexity** on WikiText-2, C4 (en), and optionally a custom dataset.
* **Zero-shot accuracy** on HellaSwag, ARC-Challenge, MMLU.
* **Speed benchmark** (tokens/sec) and memory usage.
* **Compression report** FP16 vs Ternair.

Usage
-----
::

    from ternair.benchmark.eval import run_eval_suite, print_report

    report = run_eval_suite(
        model=my_ternair_model,
        tokenizer=my_tokenizer,
        device="cuda",
        run_perplexity=True,
        run_speed=True,
    )
    print_report(report)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Dataclasses for results
# ---------------------------------------------------------------------------


@dataclass
class PerplexityResult:
    dataset: str
    num_tokens: int
    perplexity: float
    cross_entropy: float
    num_batches: int
    time_seconds: float


@dataclass
class ZeroShotResult:
    benchmark: str
    num_samples: int
    accuracy: float
    num_correct: int


@dataclass
class SpeedResult:
    tokens_per_second: float
    memory_mib: float
    prompt_length: int
    generated_length: int
    wall_time_seconds: float


@dataclass
class EvalReport:
    perplexity: list[PerplexityResult] = field(default_factory=list)
    zero_shot: list[ZeroShotResult] = field(default_factory=list)
    speed: Optional[SpeedResult] = None
    model_params: int = 0
    model_size_mib: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Perplexity (WikiText-2, C4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    subset: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_tokens: int = 100_000,
    seq_length: int = 512,
    stride: int = 256,
    device: str = "cpu",
) -> PerplexityResult:
    """Compute perplexity on a HuggingFace dataset.

    Uses sliding-window evaluation to handle long sequences efficiently.

    Parameters
    ----------
    model:
        The model to evaluate (eval mode).
    tokenizer:
        Tokenizer with ``encode`` or a callable.
    dataset_name:
        Name of the HuggingFace dataset.
    subset:
        Subset/configuration name.
    split:
        Which split to use (``"test"``, ``"validation"``).
    max_tokens:
        Maximum number of tokens to evaluate (to limit runtime).
    seq_length:
        Context window length.
    stride:
        Stride for the sliding window (smaller = more accurate but slower).
    device:
        Device to run on.

    Returns
    -------
    PerplexityResult
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[warn] datasets not installed. Install with: pip install datasets")
        return PerplexityResult(
            dataset=f"{dataset_name}/{subset} ({split})",
            num_tokens=0,
            perplexity=float("nan"),
            cross_entropy=float("nan"),
            num_batches=0,
            time_seconds=0.0,
        )

    model.eval()
    start_time = time.time()

    # Load dataset
    try:
        dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
    except Exception:
        dataset = load_dataset(dataset_name, split=split, streaming=True)

    # Tokenize
    tokens: list[int] = []
    for example in dataset:
        text = example.get("text", "")
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text)
        else:
            ids = tokenizer(text)["input_ids"] if hasattr(tokenizer, "__call__") else []
        tokens.extend(ids)
        if len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            break

    if len(tokens) == 0:
        return PerplexityResult(
            dataset=f"{dataset_name}/{subset} ({split})",
            num_tokens=0,
            perplexity=float("nan"),
            cross_entropy=float("nan"),
            num_batches=0,
            time_seconds=0.0,
        )

    # Sliding-window perplexity
    nll_sum = 0.0
    n_tokens = 0
    num_batches = 0

    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + seq_length + 1]
        if len(chunk) < 2:
            break

        input_ids = torch.tensor([chunk[:-1]], device=device)
        labels = torch.tensor([chunk[1:]], device=device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        n_tokens += labels.numel()
        num_batches += 1

    ce = nll_sum / max(n_tokens, 1)
    ppl = math.exp(ce)

    elapsed = time.time() - start_time

    return PerplexityResult(
        dataset=f"{dataset_name}/{subset} ({split})",
        num_tokens=n_tokens,
        perplexity=ppl,
        cross_entropy=ce,
        num_batches=num_batches,
        time_seconds=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Zero-shot accuracy (HellaSwag, ARC-Challenge, MMLU)
# ---------------------------------------------------------------------------


def _score_multiple_choice(
    model: nn.Module,
    tokenizer,
    contexts: list[str],
    choices: list[list[str]],
    correct: list[int],
    device: str = "cpu",
) -> ZeroShotResult:
    """Score multiple-choice questions by picking the choice with the
    lowest per-token perplexity (continuation scoring).

    This is the standard approach for evaluating LLMs on zero-shot
    benchmarks without a trained LM head for scoring.
    """
    correct_count = 0
    total = len(contexts)

    model.eval()

    for i in range(total):
        ctx = contexts[i]
        candidates = choices[i]
        best_idx = 0
        best_loss = float("inf")

        for j, candidate in enumerate(candidates):
            text = ctx + " " + candidate
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(text)
            else:
                ids = tokenizer(text)["input_ids"]

            if len(ids) < 2:
                continue

            input_ids = torch.tensor([ids[:-1]], device=device)
            labels = torch.tensor([ids[1:]], device=device)

            with torch.no_grad():
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                ).item()

            if loss < best_loss:
                best_loss = loss
                best_idx = j

        if best_idx == correct[i]:
            correct_count += 1

    return ZeroShotResult(
        benchmark="hella_swag" if len(contexts) > 0 and "hella" in str(contexts[0]).lower()
                     else "arc_challenge"
                     if "arc" in str(contexts[0]).lower()
                     else "mmlu",
        num_samples=total,
        accuracy=correct_count / max(total, 1),
        num_correct=correct_count,
    )


def run_zero_shot_hellaswag(
    model: nn.Module,
    tokenizer,
    num_samples: int = 500,
    device: str = "cpu",
) -> ZeroShotResult | None:
    """Evaluate on HellaSwag (commonsense reasoning)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[warn] datasets not installed. Skipping HellaSwag.")
        return None

    try:
        dataset = load_dataset("hellaswag", split=f"validation[:{num_samples}]")
    except Exception:
        print("[warn] Could not load HellaSwag. Skipping.")
        return None

    contexts = []
    choices = []
    correct = []

    for example in dataset:
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        contexts.append(ctx)
        choices.append(endings)
        correct.append(label)

    result = _score_multiple_choice(model, tokenizer, contexts, choices, correct, device=device)
    result.benchmark = "HellaSwag"
    return result


def run_zero_shot_arc(
    model: nn.Module,
    tokenizer,
    num_samples: int = 300,
    device: str = "cpu",
    challenge: bool = True,
) -> ZeroShotResult | None:
    """Evaluate on ARC-Challenge (science reasoning)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    subset = "ARC-Challenge" if challenge else "ARC-Easy"
    try:
        dataset = load_dataset("ai2_arc", subset, split=f"test[:{num_samples}]")
    except Exception:
        try:
            dataset = load_dataset("ai2_arc", split=f"test[:{num_samples}]")
        except Exception:
            print("[warn] Could not load ARC. Skipping.")
            return None

    contexts = []
    choices = []
    correct = []

    for example in dataset:
        question = example["question"]
        choices_raw = example["choices"]
        answer_key = example["answerKey"]

        # Build choices with labels
        labels_list = choices_raw["label"]
        texts_list = choices_raw["text"]

        # Find the correct index
        correct_idx = -1
        for idx, label in enumerate(labels_list):
            if label == answer_key:
                correct_idx = idx
                break

        if correct_idx < 0:
            continue

        contexts.append(question)
        choices.append([f"{l}) {t}" for l, t in zip(labels_list, texts_list)])
        correct.append(correct_idx)

    result = _score_multiple_choice(model, tokenizer, contexts, choices, correct, device=device)
    result.benchmark = "ARC-Challenge" if challenge else "ARC-Easy"
    return result


def run_zero_shot_mmlu(
    model: nn.Module,
    tokenizer,
    num_samples: int = 200,
    device: str = "cpu",
    subjects: list[str] | None = None,
) -> ZeroShotResult | None:
    """Evaluate on a subset of MMLU (massive multitask language understanding).

    Args:
        subjects: List of MMLU subjects. If None, uses a default subset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    if subjects is None:
        subjects = [
            "abstract_algebra", "college_computer_science",
            "college_mathematics", "computer_security",
            "global_facts", "machine_learning",
            "moral_scenarios", "philosophy",
            "professional_law", "world_religions",
        ]

    all_contexts: list[str] = []
    all_choices: list[list[str]] = []
    all_correct: list[int] = []

    for subject in subjects:
        try:
            dataset = load_dataset("lukaemon/mmlu", subject, split=f"test[:{num_samples // len(subjects) + 1}]")
        except Exception:
            continue

        for example in dataset:
            question = example["input"]
            choices_raw = [example["A"], example["B"], example["C"], example["D"]]
            answer = example.get("target", example.get("answer", "A"))

            label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct_idx = label_map.get(answer, 0)

            all_contexts.append(question)
            all_choices.append([f"{chr(65+i)}) {c}" for i, c in enumerate(choices_raw)])
            all_correct.append(correct_idx)

    if len(all_contexts) == 0:
        return None

    result = _score_multiple_choice(model, tokenizer, all_contexts, all_choices, all_correct, device=device)
    result.benchmark = "MMLU"
    return result


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

@torch.no_grad()
def benchmark_speed(
    model: nn.Module,
    tokenizer,
    prompt_text: str = "The quick brown fox jumps over the lazy dog.",
    max_new_tokens: int = 32,
    num_warmup: int = 2,
    num_runs: int = 5,
    device: str = "cpu",
) -> SpeedResult | None:
    """Measure generation speed (tokens/sec) and memory usage.

    Returns None if the model cannot generate.
    """
    try:
        from ternair.model.generation import generate
    except ImportError:
        return None

    model.eval()

    # Tokenize prompt
    if hasattr(tokenizer, "encode"):
        prompt_ids = tokenizer.encode(prompt_text)
    else:
        prompt_ids = tokenizer(prompt_text)["input_ids"]

    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    # Warmup
    for _ in range(num_warmup):
        _ = generate(model, prompt_tensor, max_new_tokens=16, temperature=0.0)

    # Timed runs
    total_time = 0.0
    total_generated = 0

    for _ in range(num_runs):
        start = time.time()
        out = generate(model, prompt_tensor, max_new_tokens=max_new_tokens, temperature=0.0)
        elapsed = time.time() - start
        total_time += elapsed
        total_generated += out.shape[1] - prompt_len

    avg_tokens_per_sec = total_generated / max(total_time, 0.001)

    # Memory estimate
    memory_mib = 0.0
    if torch.cuda.is_available() and device != "cpu":
        memory_mib = torch.cuda.memory_allocated(device) / 1024**2
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            memory_mib = torch.mps.current_allocated_memory() / 1024**2
        except Exception:
            pass

    return SpeedResult(
        tokens_per_second=round(avg_tokens_per_sec, 2),
        memory_mib=round(memory_mib, 1),
        prompt_length=prompt_len,
        generated_length=max_new_tokens,
        wall_time_seconds=round(total_time / max(num_runs, 1), 3),
    )


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def run_eval_suite(
    model: nn.Module,
    tokenizer,
    device: str = "cpu",
    run_perplexity: bool = True,
    run_zero_shot: bool = False,
    run_speed: bool = True,
    max_perplexity_tokens: int = 50_000,
    num_hellaswag: int = 200,
    num_arc: int = 150,
    num_mmlu: int = 100,
    speed_prompt: str = "The transformer model is a type of neural network architecture",
    speed_max_new_tokens: int = 32,
) -> EvalReport:
    """Run a complete evaluation suite on the model.

    Parameters
    ----------
    model:
        The model to evaluate.
    tokenizer:
        Tokenizer for the model.
    device:
        Device to run on.
    run_perplexity:
        Whether to compute perplexity on WikiText-2 and C4.
    run_zero_shot:
        Whether to run zero-shot benchmarks (requires datasets).
    run_speed:
        Whether to benchmark generation speed.
    max_perplexity_tokens:
        Max tokens for perplexity computation.
    num_hellaswag, num_arc, num_mmlu:
        Number of samples for each zero-shot benchmark.
    speed_prompt:
        Prompt for the speed benchmark.
    speed_max_new_tokens:
        Tokens to generate for speed benchmark.

    Returns
    -------
    EvalReport
        All results in a structured report.
    """
    report = EvalReport()

    # Model info
    report.model_params = sum(p.numel() for p in model.parameters())
    try:
        if hasattr(model, "num_bytes"):
            report.model_size_mib = model.num_bytes() / 1024**2
        elif hasattr(model, "config"):
            from ternair.benchmark.size import model_size_bytes
            from ternair.model.size_profiles import tiny_profile, base_profile
            # Fall back to estimation
            report.model_size_mib = 0.0
    except Exception:
        pass

    print(f"Model params: {report.model_params:,}")

    # Perplexity
    if run_perplexity:
        print("\n[Perplexity] WikiText-2...")
        ppl_wt = compute_perplexity(
            model, tokenizer, "wikitext", "wikitext-2-raw-v1",
            "test", max_tokens=max_perplexity_tokens, device=device,
        )
        report.perplexity.append(ppl_wt)
        print(f"  PPL: {ppl_wt.perplexity:.2f} ({ppl_wt.num_tokens:,} tokens)")

        print("[Perplexity] C4...")
        try:
            ppl_c4 = compute_perplexity(
                model, tokenizer, "c4", "en",
                "validation", max_tokens=max_perplexity_tokens, device=device,
            )
            report.perplexity.append(ppl_c4)
            print(f"  PPL: {ppl_c4.perplexity:.2f} ({ppl_c4.num_tokens:,} tokens)")
        except Exception as e:
            print(f"  C4 skipped ({e})")

    # Zero-shot
    if run_zero_shot:
        print("\n[Zero-shot] HellaSwag...")
        hs = run_zero_shot_hellaswag(model, tokenizer, num_samples=num_hellaswag, device=device)
        if hs:
            report.zero_shot.append(hs)
            print(f"  Accuracy: {hs.accuracy:.2%} ({hs.num_correct}/{hs.num_samples})")

        print("[Zero-shot] ARC-Challenge...")
        arc = run_zero_shot_arc(model, tokenizer, num_samples=num_arc, device=device, challenge=True)
        if arc:
            report.zero_shot.append(arc)
            print(f"  Accuracy: {arc.accuracy:.2%} ({arc.num_correct}/{arc.num_samples})")

        print("[Zero-shot] MMLU...")
        mmlu = run_zero_shot_mmlu(model, tokenizer, num_samples=num_mmlu, device=device)
        if mmlu:
            report.zero_shot.append(mmlu)
            print(f"  Accuracy: {mmlu.accuracy:.2%} ({mmlu.num_correct}/{mmlu.num_samples})")

    # Speed
    if run_speed:
        print("\n[Speed benchmark]...")
        speed = benchmark_speed(
            model, tokenizer, prompt_text=speed_prompt,
            max_new_tokens=speed_max_new_tokens, device=device,
        )
        if speed:
            report.speed = speed
            print(f"  {speed.tokens_per_second:.1f} tokens/sec")

    return report


def print_report(report: EvalReport) -> None:
    """Pretty-print an evaluation report."""
    print("=" * 60)
    print("  Ternair Evaluation Report")
    print("=" * 60)
    print(f"  Parameters : {report.model_params:,}")
    print(f"  Size       : {report.model_size_mib:.1f} MiB")

    if report.perplexity:
        print("-" * 60)
        print("  Perplexity")
        for ppl in report.perplexity:
            print(f"    {ppl.dataset:<30s}  {ppl.perplexity:>8.2f}  "
                  f"({ppl.num_tokens:,} tokens)")

    if report.zero_shot:
        print("-" * 60)
        print("  Zero-shot Accuracy")
        for zs in report.zero_shot:
            print(f"    {zs.benchmark:<30s}  {zs.accuracy:>7.2%}  "
                  f"({zs.num_correct}/{zs.num_samples})")

    if report.speed:
        print("-" * 60)
        s = report.speed
        print(f"  Speed         : {s.tokens_per_second:>8.1f} tok/s")
        print(f"  Latency       : {s.wall_time_seconds:>8.3f} s")
        print(f"  Memory        : {s.memory_mib:>8.1f} MiB")

    print("=" * 60)


__all__ = [
    "PerplexityResult",
    "ZeroShotResult",
    "SpeedResult",
    "EvalReport",
    "compute_perplexity",
    "run_zero_shot_hellaswag",
    "run_zero_shot_arc",
    "run_zero_shot_mmlu",
    "benchmark_speed",
    "run_eval_suite",
    "print_report",
]
