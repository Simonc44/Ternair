"""Benchmark utilities (size analyser + evaluation suite)."""

from ternair.benchmark.size import (
    SizeBreakdown,
    describe,
    model_size_bytes,
    auto_fit_to_bytes,
    fit_one_gb,
)
from ternair.benchmark.eval import (
    PerplexityResult,
    ZeroShotResult,
    SpeedResult,
    EvalReport,
    compute_perplexity,
    run_zero_shot_hellaswag,
    run_zero_shot_arc,
    run_zero_shot_mmlu,
    benchmark_speed,
    run_eval_suite,
    print_report,
)

__all__ = [
    "SizeBreakdown",
    "describe",
    "model_size_bytes",
    "auto_fit_to_bytes",
    "fit_one_gb",
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
