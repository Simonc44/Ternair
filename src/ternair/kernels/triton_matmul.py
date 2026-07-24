"""Backward-compatibility shim -- re-exports from ``ternair.kernels.triton_fast``.

The canonical implementation lives in
:mod:`ternair.kernels.triton_fast` since v0.6.0.  This module is kept
so that ``from ternair.kernels.triton_matmul import ...`` continues
to work for users on v0.5.0 and earlier.

The legacy single-batch function ``ternary_matmul_triton`` accepted a
``(N,)`` input and returned a ``(M,)`` output.  The new canonical
``ternary_matmul_triton`` accepts both ``(N,)`` and ``(B, N)`` and
returns the matching shape -- so existing single-batch callers see
the same signature behaviour.
"""

from __future__ import annotations

import warnings

from ternair.kernels.triton_fast import (  # noqa: F401
    benchmark_triton_vs_numpy,
    has_triton,
    ternary_matmul_single_triton,
    ternary_matmul_triton,
)

warnings.warn(
    "Importing from 'ternair.kernels.triton_matmul' is deprecated since v0.6.0; "
    "use 'ternair.kernels.triton_fast' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "has_triton",
    "ternary_matmul_triton",
    "ternary_matmul_single_triton",
    "benchmark_triton_vs_numpy",
]
