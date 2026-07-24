"""Triton (GPU) kernel for the packed ternary matmul.

The kernel decodes 2‑bit fastpacked bytes directly inside Triton's
JIT-compiled loop and accumulates in FP32 — **no temporary float
weight tensor** is ever materialised.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

_LOGGER = logging.getLogger(__name__)

# Lazy-loaded; False on import failure.
_TRITON_KERNEL = None
_HAS_TRITON = None


def has_triton() -> bool:
    global _HAS_TRITON
    if _HAS_TRITON is None:
        try:
            import torch  # noqa: F401
            import triton  # noqa: F401

            _HAS_TRITON = True
        except Exception:
            _HAS_TRITON = False
    return bool(_HAS_TRITON)


def _compile_kernel() -> bool:
    """Try to import Triton and produce the JIT kernel.  Returns True on success."""
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return bool(_TRITON_KERNEL)
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        @triton.jit  # type: ignore[no-redef]
        def _ternary_fastpacked_kernel(
            packed_ptr,  # uint8  (M, Kp)
            x_ptr,  # float16  (N,)
            gamma_ptr,  # float32  (M,)
            out_ptr,  # float16  (M,)
            stride_pm,
            stride_pk,
            stride_x: tl.constexpr,
            stride_gamma: tl.constexpr,
            M: tl.constexpr,
            N: tl.constexpr,
            K_PACKED: tl.constexpr,
            BLOCK_M: tl.constexpr,
        ):
            pid = tl.program_id(0)
            m_start = pid * BLOCK_M
            m_offs = m_start + tl.arange(0, BLOCK_M)
            m_mask = m_offs < M

            acc = tl.zeros([BLOCK_M], dtype=tl.float32)

            for k in range(K_PACKED):
                byte = tl.load(
                    packed_ptr + m_offs * stride_pm + k * stride_pk,
                    mask=m_mask,
                    other=tl.uint8(0),
                )
                for pos in tl.static_range(4):
                    n = k * 4 + pos
                    if n < N:
                        bits = (byte >> (2 * pos)) & 3
                        trit = (bits & 1) - ((bits >> 1) & 1)
                        x_val = tl.load(x_ptr + n * stride_x)
                        acc += trit.to(tl.float32) * x_val

            gamma = tl.load(gamma_ptr + m_offs * stride_gamma, mask=m_mask, other=0.0)
            out = acc * gamma
            tl.store(out_ptr + m_offs, out.to(tl.float16), mask=m_mask)

        _TRITON_KERNEL = _ternary_fastpacked_kernel
        return True
    except Exception as exc:
        _LOGGER.info("Triton kernel not compiled (%s)", exc)
        _TRITON_KERNEL = False
        return False


def ternary_matmul_triton(
    packed: NDArray[np.uint8],
    x: NDArray[np.float16],
    gamma: NDArray[np.float32],
    *,
    device: str = "cuda",
    block_m: int = 32,
) -> NDArray[np.float16]:
    """Compute ``y[m] = γ[m] · sum_n(W_t[m,n] · x[n])`` on GPU.

    Falls back to the numpy reference when Triton is unavailable.
    """
    if not _compile_kernel():
        _LOGGER.warning("Triton kernel unavailable — numpy fallback")
        from ternair.kernels.packed_ops import ternary_matmul_numpy

        return ternary_matmul_numpy(packed, x, gamma)

    import torch
    import triton  # type: ignore[import-untyped]

    M, Kp = packed.shape
    N = x.shape[0]

    packed_t = torch.from_numpy(packed).to(device)
    x_t = torch.from_numpy(x).to(device)
    gamma_t = torch.from_numpy(gamma).to(device)
    out_t = torch.empty(M, dtype=torch.float16, device=device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)

    assert _TRITON_KERNEL is not None
    _TRITON_KERNEL[grid](
        packed_t.data_ptr(),
        x_t.data_ptr(),
        gamma_t.data_ptr(),
        out_t.data_ptr(),
        packed_t.stride(0),
        packed_t.stride(1),
        x_t.stride(0),
        gamma_t.stride(0),
        M=M,
        N=N,
        K_PACKED=Kp,
        BLOCK_M=block_m,
    )

    return out_t.cpu().numpy()


__all__ = ["has_triton", "ternary_matmul_triton"]
