"""Python integration test for the native C++ engine.

Skipped automatically if ``libternair_native.so`` is not built.
Verifies that the ctypes wrapper produces the same matmul output as the
NumPy reference (within fp16 noise).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def _try_load():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "python"))
    from native import _find_library
    return _find_library()


def main() -> int:
    lib = _try_load()
    if lib is None:
        print("[test_native] SKIP (libternair_native.so not built -- run scripts/build.sh)")
        return 0

    from native import ternary_matmul
    from ternair.kernels.packing_fast import pack_trits_2bit
    from ternair.kernels.packed_ops import ternary_matmul_numpy_batched

    rng = np.random.default_rng(7)
    M, N = 64, 128   # multiples of 4
    Kp = (N + 3) // 4
    trits = rng.integers(-1, 2, size=(M * N,)).astype(np.int8)
    packed = pack_trits_2bit(trits).reshape(M, Kp).astype(np.uint8)
    gamma = rng.random(M).astype(np.float32)
    x = rng.random((1, N)).astype(np.float16)
    x_view = x.view(np.uint16).reshape(-1)

    # Native
    t0 = time.perf_counter()
    out_native_bits = ternary_matmul(packed, x_view, gamma)
    t_native = time.perf_counter() - t0

    # NumPy reference
    t0 = time.perf_counter()
    out_numpy = ternary_matmul_numpy_batched(packed, x, gamma)
    t_numpy = time.perf_counter() - t0

    out_native = out_native_bits.view(np.float16).astype(np.float32)
    out_numpy_f = out_numpy.astype(np.float32)
    diff = float(np.abs(out_native - out_numpy_f).max())
    mag = float(np.maximum(np.abs(out_native).mean(), np.abs(out_numpy_f).mean(), 1e-9))
    rel = diff / mag

    print(f"[test_native] M={M} N={N} backend=native")
    print(f"[test_native] numpy mean={t_native*1e3:.3f} ms  native mean={t_numpy*1e3:.3f} ms")
    print(f"[test_native] max|err|={diff:.4g}  rel={rel:.4g}")

    if rel < 0.05:
        print("[test_native] PASS")
        return 0
    print("[test_native] FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
