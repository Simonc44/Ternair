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

Usage (numpy, CPU fallback)::

    from ternair.kernels.triton_fast import ternary_matmul_triton
    y = ternary_matmul_triton(packed, x, gamma, device="cuda")
    # x: (B, N) float16, packed: (M, ceil(N/4)) uint8, gamma: (M,) float32
    # returns: (B, M) float16

Usage (torch, real GPU)::

    # No GPU->CPU->GPU roundtrip -- tensors are moved to CUDA in-place.
    out = ternary_matmul_triton(packed_uint8, x_fp16, gamma_fp32,
                                device="cuda")   # returns torch.Tensor on CUDA
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


# ---------------------------------------------------------------------------
# Mixed-input helpers
# ---------------------------------------------------------------------------


def _is_torch_tensor(x: object) -> bool:
    """True if ``x`` is a torch.Tensor (import lazily to avoid hard dep)."""
    if not hasattr(x, "detach") or not hasattr(x, "cpu"):
        return False
    try:
        import torch  # noqa: F401

        return isinstance(x, torch.Tensor)
    except Exception:
        return False


def _to_numpy(x: object) -> NDArray:
    """Coerce torch.Tensor OR numpy array to a contiguous numpy array.

    Triggers a GPU->CPU sync if ``x`` is a CUDA tensor -- callers that
    care about staying on the GPU should use :func:`_to_torch` instead.
    """
    if x is None:
        raise ValueError("Got None tensor")
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x)
    if _is_torch_tensor(x):
        return np.ascontiguousarray(x.detach().cpu().numpy())
    raise TypeError(f"Unsupported tensor type: {type(x).__name__}")


def _to_torch(x: object, *, device: str, dtype) -> "torch.Tensor":
    """Move a torch.Tensor / numpy array to the given device + dtype."""
    import torch

    if _is_torch_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Main kernel entry-point (mixed torch/numpy)
# ---------------------------------------------------------------------------


def ternary_matmul_triton(
    packed,
    x,
    gamma,
    *,
    device: str = "cuda",
    block_m: int = 64,
):
    """Compute ``y = gamma * (W_t @ x.T).T`` on GPU or via fallback.

    Parameters
    ----------
    packed
        ``(M, K_packed)`` ``uint8`` -- fastpacked weights
        (4 trits per byte, ``K_packed = ceil(N / 4)``).
    x
        ``(B, N)`` (or (N,)) float16 -- input activations.
        ``np.ndarray`` or ``torch.Tensor`` (with a CUDA copy when Triton
        is active, no sync back).
    gamma
        ``(M,)`` float32 -- per-output scaling factor.
    device
        Target device (default ``"cuda"``).
    block_m
        Row-block size per program (default 64).

    Returns
    -------
    y
        ``(B, M)`` float16.  Returns a numpy array when the NumPy
        fallback path runs, a torch.Tensor (on ``device``) when the
        Triton path runs and at least one input was a torch.Tensor;
        otherwise the result is moved back to the original layout.
    """
    # ------------------------------------------------------------------
    # Decide which path: torch-Triton OR numpy-fallback
    # ------------------------------------------------------------------
    use_triton = bool(has_triton() and str(device).startswith("cuda"))
    input_is_torch = _is_torch_tensor(x) or _is_torch_tensor(packed) or _is_torch_tensor(gamma)

    if not use_triton:
        # Use NumPy fallback always.
        return _numpy_fallback(packed, x, gamma)

    # ------------------------------------------------------------------
    # Triton path -- do NOT GPU->CPU->GPU : keep torch tensors on CUDA.
    # ------------------------------------------------------------------
    import torch
    import triton

    # Coerce / promote
    packed_t = _to_torch(packed, device=device, dtype=torch.uint8)
    gamma_t = _to_torch(gamma, device=device, dtype=torch.float32)
    # x may be (N,) or (B, N) -- promote to 2-D and remember if we squeezed.
    if _is_torch_tensor(x):
        if x.dim() == 1:
            x_t = x.to(device=device, dtype=torch.float16).unsqueeze(0)
            squeezed = True
        else:
            x_t = x.to(device=device, dtype=torch.float16)
            squeezed = False
        out_dtype = torch.float16
        out_device = device
    else:
        if x.ndim == 1:
            x_t = (_to_torch(x, device=device, dtype=torch.float16)).unsqueeze(0)
            squeezed = True
        else:
            x_t = _to_torch(x, device=device, dtype=torch.float16)
            squeezed = False
        out_dtype = torch.float16
        out_device = device

    if x_t.dim() != 2:
        raise ValueError(f"x must be 1-D or 2-D, got shape {tuple(x_t.shape)}")

    B, N = x_t.shape
    M, K_packed = packed_t.shape
    if K_packed != (N + 3) // 4:
        raise ValueError(
            f"packed.shape[1]={K_packed} inconsistent with N={N} "
            f"(expected ceil(N/4)={(N + 3) // 4})"
        )
    if gamma_t.shape != (M,):
        raise ValueError(f"gamma must be (M,), got {tuple(gamma_t.shape)}")

    # Lazy-compile the kernel once.
    _compile_kernel()

    out_t = torch.empty((B, M), dtype=out_dtype, device=out_device)
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

    # If the user passed pure numpy inputs, move the result back to numpy.
    if not input_is_torch:
        out = out_t.cpu().numpy()
        if squeezed:
            out = out.squeeze(0)
        return out

    if squeezed:
        return out_t.squeeze(0)
    return out_t


# ---------------------------------------------------------------------------
# NumPy fallback path
# ---------------------------------------------------------------------------


def _numpy_fallback(packed, x, gamma):
    from ternair.kernels.packed_ops import ternary_matmul_numpy_batched

    packed_np = _to_numpy(packed)
    x_np = _to_numpy(x).astype(np.float16, copy=False)
    gamma_np = _to_numpy(gamma).astype(np.float32, copy=False)

    if x_np.ndim == 1:
        x_np = x_np[np.newaxis, :]
        squeezed = True
    else:
        squeezed = False

    out = ternary_matmul_numpy_batched(packed_np, x_np, gamma_np)
    if squeezed:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Single-batch convenience wrapper (kept for backward compat with code
# that explicitly expects a 1-D output).  Implemented in terms of the
# unified kernel above.
# ---------------------------------------------------------------------------


def ternary_matmul_single_triton(
    packed,
    x,
    gamma,
    *,
    device: str = "cuda",
    block_m: int = 64,
):
    """Backward-compat wrapper: ``(N,)`` -> ``(M,)`` output."""
    out = ternary_matmul_triton(packed, x, gamma, device=device, block_m=block_m)
    if hasattr(out, "ndim") and out.ndim == 2 and out.shape[0] == 1:
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
