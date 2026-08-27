"""Micro‑benchmark for the packed ternary matmul backends.

Usage
-----
.. code-block:: bash

    PYTHONPATH=src python -m ternair.kernels.benchmark [--size MxN] [--trials 5]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ternair.kernels.packed_ops import ternary_matmul_numpy, decode_fastpacked_row
from ternair.kernels.packing_fast import pack_trits_2bit


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make_random_weights(M: int, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(packed, gamma, trits)`` with randomly generated weights."""
    rng = _rng()
    trits = rng.choice([-1, 0, 1], size=(M, N), p=[0.25, 0.50, 0.25]).astype(np.int8)
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])
    gamma = rng.random(M).astype(np.float32)
    return packed, gamma, trits


def run_benchmark(
    M: int, N: int, batch: int = 1, trials: int = 5
) -> dict[str, float]:
    """Benchmark each available backend and return ``{name: median_ms}``."""
    packed, gamma, _ = _make_random_weights(M, N)
    if batch > 1:
        x = _rng().random((batch, N)).astype(np.float16)
    else:
        x = _rng().random(N).astype(np.float16)

    results: dict[str, float] = {}

    # 1) numpy reference
    elapsed = []
    for _ in range(trials):
        t0 = time.perf_counter()
        if batch > 1:
            from ternair.kernels.packed_ops import ternary_matmul_numpy_batched
            _ = ternary_matmul_numpy_batched(packed, x, gamma)
        else:
            _ = ternary_matmul_numpy(packed, x, gamma)
        elapsed.append(time.perf_counter() - t0)
    results["numpy"] = float(np.median(elapsed) * 1000)  # ms

    # 2) Triton (GPU) — if available
    from ternair.kernels.triton_matmul import has_triton, ternary_matmul_triton
    if has_triton() and batch == 1:
        elapsed = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = ternary_matmul_triton(packed, x, gamma)
            elapsed.append(time.perf_counter() - t0)
        results["triton"] = float(np.median(elapsed) * 1000)
    else:
        results["triton"] = float("nan")

    # 3) C++ CPU backend — if available
    from ternair.kernels.cpu_matmul import has_cpu_backend, ternary_matmul_cpp
    if has_cpu_backend() and batch == 1:
        elapsed = []
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = ternary_matmul_cpp(packed, x, gamma)
            elapsed.append(time.perf_counter() - t0)
        results["cpp"] = float(np.median(elapsed) * 1000)
    else:
        results["cpp"] = float("nan")

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Packed ternary matmul benchmark")
    p.add_argument("--size", default="512x4096", help="MxN dimensions")
    p.add_argument("--trials", type=int, default=5)
    args = p.parse_args()

    M_str, N_str = args.size.lower().split("x")
    M, N = int(M_str), int(N_str)

    print(f"Benchmark:  {M}x{N}  packed ternary matmul  ({args.trials} trials)")
    print("-" * 60)

    res = run_benchmark(M, N, trials=args.trials)
    for backend, ms in res.items():
        if np.isnan(ms):
            print(f"  {backend:16s}  N/A (not available)")
        else:
            gflops = (2 * M * N) / (ms * 1e-3) / 1e9
            print(f"  {backend:16s}  {ms:6.1f} ms  ({gflops:.1f} GFLOP/s)")


if __name__ == "__main__":
    main()
