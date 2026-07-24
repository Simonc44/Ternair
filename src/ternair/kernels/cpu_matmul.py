"""Python wrapper and C++ header for CPU‑SIMD ternary matmul.

The C++ implementation provides three backends selected at runtime:

* **AVX‑512** (x86‑64, Intel / AMD) — uses masks to ADD/SUB packed
  fp16 activations.
* **ARM NEON** (Apple Silicon, ARM‑v8/v9) — uses ``vaddq_f16`` /
  ``vsubq_f16`` with ``vbslq_f16`` for masking.
* **Scalar fallback** — plain loop (always available).

The C++ code is embedded in this module both as header source
(:data:`_CPP_HEADER`) and as a compile‑on‑call extension via ``cppyy``
when available, or as a ``ctypes`` wrapper.

The user can also copy the header ``cpu_matmul.h`` out of the package
and link it into their own application.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray

_LOGGER = logging.getLogger(__name__)

# The C++ header is stored alongside this file.
_HEADER_PATH = os.path.join(os.path.dirname(__file__), "cpu_matmul.h")

_HAS_CPP_BACKEND: bool | None = None


def has_cpu_backend() -> bool:
    """``True`` if the compiled C++ backend is available."""
    global _HAS_CPP_BACKEND
    if _HAS_CPP_BACKEND is None:
        try:
            import cppyy  # noqa: F401

            _HAS_CPP_BACKEND = True
        except Exception:
            _HAS_CPP_BACKEND = False
    return _HAS_CPP_BACKEND


def _load_cpp_backend():
    if not has_cpu_backend():
        raise ImportError("cppyy is required to load the C++ backend")
    import cppyy

    with open(_HEADER_PATH) as f:
        cppyy.cppdef(f.read())


def ternary_matmul_cpp(
    packed: NDArray[np.uint8],
    x: NDArray[np.float16],
    gamma: NDArray[np.float32],
    *,
    threads: Optional[int] = None,
) -> NDArray[np.float16]:
    """Packed ternary matmul via the C++ SIMD backend.

    If ``cppyy`` is installed and the header compiles cleanly the call
    is dispatched to AVX‑512 (x86) or NEON (ARM).  Otherwise falls back
    to the numpy reference.
    """
    if not has_cpu_backend():
        _LOGGER.warning("C++ backend not available — using numpy fallback")
        from ternair.kernels.packed_ops import ternary_matmul_numpy

        return ternary_matmul_numpy(packed, x, gamma)

    try:
        _load_cpp_backend()
        import cppyy
        import cppyy.ll

        M, Kp = packed.shape
        N = x.shape[0]

        out = np.empty(M, dtype=np.float16)
        n_threads = threads or 0  # 0 → backend picks default

        cppyy.ll.call(
            "ternary_matmul_cxx_dispatch",
            packed.ctypes,
            M,
            Kp,
            x.ctypes,
            N,
            gamma.ctypes,
            out.ctypes,
            n_threads,
        )
        return out
    except Exception as exc:
        _LOGGER.warning("C++ backend error (%s) — numpy fallback", exc)
        from ternair.kernels.packed_ops import ternary_matmul_numpy

        return ternary_matmul_numpy(packed, x, gamma)


__all__ = [
    "has_cpu_backend",
    "ternary_matmul_cpp",
    "_HEADER_PATH",
]
