"""Triton GPU kernel for ternary matmul -- 4 trits per byte (fastpacked).

This module supersedes the legacy ``ternair.kernels.triton_matmul`` that
existed through v0.5.0.  The new kernel:

1. **Treats all inputs as 2-D** ``(B, N)`` -- even when ``B = 1`` --
   so the same kernel handles both single-vector and batched matmul.
2. **Uses a 2-D grid** ``(pid_m, pid_b)`` that lets each thread-block
   compute one row-block of the output for one batch element.
3. **Reads the 4 trits from each byte via bit arithmetic** (no LUT,
   no branch), so register pressure stays low and the kernel is
   portable across CUDA / ROCm / Intel GPUs.

The kernel operates directly on the fastpacked ``uint8`` buffer that
:func:`ternair.kernels.packing_fast.pack_trits_2bit` produces.  No
temporary float weight tensor is ever materialised.

Usage::

    from ternair.kernels.triton_fast import ternary_matmul_triton
    y = ternary_matmul_triton(packed, x, gamma, device="cuda")
    # x: (B, N) float16, packed: (M, ceil(N/4)) uint8, gamma: (M,) float32
    # returns: (B, M) float16
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

_LOGGER = logging.getLogger(__name__)

# Lazy global state: avoids importing torch/triton at module import time.
_HAS_TRITON: bool | None = None
_KERNEL = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def has_triton() -> bool:
    """Return ``True`` if Triton + torch + CUDA are all available.

    The kernel only runs on a real GPU, so we refuse to claim Triton is
    available on CPU-only hosts -- callers then fall back to the NumPy
    reference implementation transparently.
    """
    global _HAS_TRITON
    if _HAS_TRITON is None:
        try:
            import torch  # noqa: F401
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401

            _HAS_TRITON = bool(torch.cuda.is_available())
        except Exception:
            _HAS_TRITON = False
    return bool(_HAS_TRITON)


def ternary_matmul_triton(
    packed: NDArray[np.uint8],
    x: NDArray[np.float16],
    gamma: NDArray[np.float32],
    *,
    device: str = "cuda",
    block_m: int = 64,
) -> NDArray[np.float16]:
    """Compute ``y = gamma * (W_t @ x.T).T`` on GPU.

    Parameters
    ----------
    packed
        ``(M, K_packed)`` ``uint8`` -- fastpacked weights
        (4 trits per byte, ``K_packed = ceil(N / 4)``).
    x
        ``(B, N)`` ``float16`` -- input activations.  ``B = 1`` is OK.
    gamma
        ``(M,)`` ``float32`` -- per-output scaling factor.
    device
        Target device (default ``"cuda"``).
    block_m
        Row-block size per program (default 64).

    Returns
    -------
    y
        ``(B, M)`` ``float16`` -- output activations.
    """
    if not has_triton() or device != "cuda":
        _LOGGER.info("Triton kernel unavailable on this host; using NumPy reference")
        from ternair.kernels.packed_ops import ternary_matmul_numpy_batched

        # Ensure 2-D input for the reference.
        squeezed = False
        if x.ndim == 1:
            x = x[np.newaxis, :]
            squeezed = True
        out = ternary_matmul_numpy_batched(packed, x, gamma)
        if squeezed:
            out = out.squeeze(0)
        return out

    # 1-D inputs get promoted to 2-D ``(1, N)`` to keep the kernel simple.
    squeezed = False
    if x.ndim == 1:
        x = x[np.newaxis, :]
        squeezed = True

    if x.ndim != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2-D (M, K_packed), got shape {packed.shape}")

    B, N = x.shape
    M, K_packed = packed.shape
    if K_packed != (N + 3) // 4:
        raise ValueError(
            f"packed.shape[1]={K_packed} inconsistent with N={N} "
            f"(expected ceil(N/4)={(N + 3) // 4})"
        )
    if gamma.shape != (M,):
        raise ValueError(f"gamma must be (M,), got {gamma.shape}")

    import torch
    import triton

    # Lazy-compile the kernel once.
    _compile_kernel()

    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).to(device)
    x_t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    gamma_t = torch.from_numpy(np.ascontiguousarray(gamma)).to(device)
    out_t = torch.empty((B, M), dtype=torch.float16, device=device)

    grid = (triton.cdiv(M, block_m), B)
    assert _KERNEL is not None
    _KERNEL[grid](
        packed_t, x_t, gamma_t, out_t,
        M, N, K_packed, B,
        packed_t.stride(0), packed_t.stride(1),
        x_t.stride(0), x_t.stride(1),
        out_t.stride(0),
        BLOCK_M=block_m,
    )

    out = out_t.cpu().numpy()
    if squeezed:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Single-batch convenience wrapper (kept for backward compat with code
# that explicitly expects a 1-D output).  Implemented in terms of the
# unified kernel above.
# ---------------------------------------------------------------------------


def ternary_matmul_single_triton(
    packed: NDArray[np.uint8],
    x: NDArray[np.float16],
    gamma: NDArray[np.float32],
    *,
    device: str = "cuda",
    block_m: int = 64,
) -> NDArray[np.float16]:
    """Backward-compat wrapper: ``(N,)`` -> ``(M,)`` output."""
    out = ternary_matmul_triton(packed, x, gamma, device=device, block_m=block_m)
    if out.ndim == 2 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Internal: Triton kernel definition
# ---------------------------------------------------------------------------


def _compile_kernel() -> None:
    """Compile the Triton kernel once.  Safe to call multiple times."""
    global _KERNEL
    if _KERNEL is not None:
        return
    try:
        import torch  # noqa: F401
        import triton
        import triton.language as tl

        @triton.jit
        def _ternary_matmul_kernel(
            packed_ptr,   # uint8  (M, K_packed)
            x_ptr,        # fp16   (B, N)
            gamma_ptr,    # fp32   (M,)
            out_ptr,      # fp16   (B, M)
            M: tl.constexpr,
            N: tl.constexpr,
            K_PACKED: tl.constexpr,
            B: tl.constexpr,
            stride_pm,
            stride_pk,
            stride_xb,
            stride_xn,
            stride_ob,
            BLOCK_M: tl.constexpr,
        ):
            pid_m = tl.program_id(0)  # row block
            pid_b = tl.program_id(1)  # batch element

            m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            m_mask = m_offs < M

            x_base = x_ptr + pid_b * stride_xb
            out_base = out_ptr + pid_b * stride_ob

            acc = tl.zeros([BLOCK_M], dtype=tl.float32)

            # One byte == 4 trits.  Unrolled statically.
            for k in range(K_PACKED):
                byte = tl.load(
                    packed_ptr + m_offs * stride_pm + k * stride_pk,
                    mask=m_mask,
                    other=tl.uint8(0),
                )
                for pos in tl.static_range(4):
                    n = k * 4 + pos
                    if n < N:
                        bits = (byte >> (2 * pos)) & 0x03
                        trit = (bits & 1) - ((bits >> 1) & 1)
                        x_val = tl.load(x_base + n * stride_xn)
                        acc += trit.to(tl.float32) * x_val

            gamma = tl.load(gamma_ptr + m_offs, mask=m_mask, other=0.0)
            out = acc * gamma
            tl.store(out_base + m_offs, out.to(tl.float16), mask=m_mask)

        _KERNEL = _ternary_matmul_kernel
    except Exception as exc:  # pragma: no cover
        _LOGGER.info("Triton kernel compile failed: %s", exc)
        _KERNEL = None


# ---------------------------------------------------------------------------
# Optional: benchmark utility (mirrors the old triton_matmul_fused one).
# ---------------------------------------------------------------------------


def benchmark_triton_vs_numpy(
    M: int = 4096,
    N: int = 4096,
    batch: int = 1,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> dict:
    """Benchmark Triton kernel vs NumPy reference (4-trits-per-byte)."""
    import time

    from ternair.kernels.packed_ops import ternary_matmul_numpy_batched
    from ternair.kernels.packing_fast import pack_trits_2bit

    trits = np.random.randint(-1, 2, size=(M, N)).astype(np.int8).reshape(-1)
    packed = pack_trits_2bit(trits).reshape(M, (N + 3) // 4)
    gamma = np.random.rand(M).astype(np.float32)
    x = np.random.randn(batch, N).astype(np.float16)

    for _ in range(num_warmup):
        _ = ternary_matmul_numpy_batched(packed, x, gamma)
        if has_triton():
            _ = ternary_matmul_triton(packed, x, gamma)

    np_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = ternary_matmul_numpy_batched(packed, x, gamma)
        np_times.append(time.perf_counter() - start)

    triton_times: list[float] = []
    if has_triton():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = ternary_matmul_triton(packed, x, gamma)
            triton_times.append(time.perf_counter() - start)

    return {
        "M": M,
        "N": N,
        "batch": batch,
        "numpy_mean_ms": float(np.mean(np_times) * 1000),
        "triton_mean_ms": float(np.mean(triton_times) * 1000) if triton_times else None,
        "speedup": (
            float(np.mean(np_times) / max(np.mean(triton_times), 1e-9))
            if triton_times
            else None
        ),
    }


__all__ = [
    "has_triton",
    "ternary_matmul_triton",
    "ternary_matmul_single_triton",
    "benchmark_triton_vs_numpy",
]
