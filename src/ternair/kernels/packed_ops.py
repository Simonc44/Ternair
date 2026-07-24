"""Pure‑Python / NumPy reference for the packed ternary matmul.

These functions show the *algorithm* that the GPU and CPU SIMD kernels
implement.  They are intentionally simple and readable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ternair.kernels.packing_fast import TRIT_LUT


def decode_fastpacked_row(packed_row: NDArray[np.uint8], N: int) -> NDArray[np.int8]:
    """Decode a single packed fastpacked row to ``{-1, 0, +1}``.

    Equivalent to :func:`~ternair.kernels.packing_fast.unpack_trits_2bit`
    but uses the LUT for maximum Python speed.
    """
    Kp = len(packed_row)
    trits = np.zeros(N, dtype=np.int8)
    for kp in range(Kp):
        byte = int(packed_row[kp])
        t = TRIT_LUT[byte]  # shape (4,)
        offset = kp * 4
        remaining = N - offset
        trits[offset: offset + min(4, remaining)] = t[:min(4, remaining)]
    return trits


def ternary_matmul_numpy(
    packed: NDArray[np.uint8],
    x: NDArray[np.float16],
    gamma: NDArray[np.float32],
) -> NDArray[np.float16]:
    """One‑off ``y = γ · (W_t · x)``  —  single batch, CPU‑only.

    Parameters
    ----------
    packed : (M, ceil(N/4)) uint8
        Fastpacked weight rows (4 trits / byte).
    x : (N,) float16
        Input activation vector.
    gamma : (M,) float32
        Per‑output scaling factor.

    Returns
    -------
    y : (M,) float16
        Output activations.
    """
    M = packed.shape[0]
    N = len(x)
    Kp = packed.shape[1]
    out = np.zeros(M, dtype=np.float32)
    for m in range(M):
        acc = 0.0
        for kp in range(Kp):
            byte = int(packed[m, kp])
            t0, t1, t2, t3 = TRIT_LUT[byte].tolist()
            n0 = kp * 4
            if t0:
                acc += t0 * float(x[n0])
            if t1 and n0 + 1 < N:
                acc += t1 * float(x[n0 + 1])
            if t2 and n0 + 2 < N:
                acc += t2 * float(x[n0 + 2])
            if t3 and n0 + 3 < N:
                acc += t3 * float(x[n0 + 3])
        out[m] = acc * float(gamma[m])
    return out.astype(np.float16)


def ternary_matmul_numpy_batched(
    packed: NDArray[np.uint8],
    x_batch: NDArray[np.float16],
    gamma: NDArray[np.float32],
) -> NDArray[np.float16]:
    """Batched version — ``(B, N)`` inputs, ``(B, M)`` outputs.

    For a transformer *per‑token* the batch dimension is the sequence
    length.  We loop over the batch for readability; Triton will fuse.
    """
    B, N = x_batch.shape
    M = packed.shape[0]
    Kp = packed.shape[1]
    out = np.zeros((B, M), dtype=np.float32)
    for b in range(B):
        x = x_batch[b]
        for m in range(M):
            acc = 0.0
            for kp in range(Kp):
                byte = int(packed[m, kp])
                t = TRIT_LUT[byte]
                n0 = kp * 4
                if t[0]:
                    acc += t[0] * float(x[n0])
                for p in range(1, 4):
                    if t[p] and n0 + p < N:
                        acc += t[p] * float(x[n0 + p])
            out[b, m] = acc * float(gamma[m])
    return out.astype(np.float16)


__all__ = [
    "decode_fastpacked_row",
    "ternary_matmul_numpy",
    "ternary_matmul_numpy_batched",
]
