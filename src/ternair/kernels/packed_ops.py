"""Pure-Python / NumPy reference for the packed ternary matmul.

These functions show the *algorithm* that the GPU and CPU SIMD kernels
implement.  The implementation is fully vectorised: the packed bytes
are decoded into the ``{-1, 0, +1}`` weight matrix once (via the 256-entry
LUT), then the output is a single ``np.matmul`` call.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ternair.kernels.packing_fast import TRIT_LUT


def decode_fastpacked_row(packed_row: NDArray[np.uint8], N: int) -> NDArray[np.int8]:
    """Decode a single packed fastpacked row to ``{-1, 0, +1}``."""
    Kp = len(packed_row)
    trits = np.zeros(N, dtype=np.int8)
    for kp in range(Kp):
        byte = int(packed_row[kp])
        t = TRIT_LUT[byte]  # shape (4,)
        offset = kp * 4
        remaining = N - offset
        trits[offset: offset + min(4, remaining)] = t[:min(4, remaining)]
    return trits


def unpack_fastpacked_matrix(packed: NDArray[np.uint8]) -> NDArray[np.int8]:
    """Decode an ``(M, Kp)`` packed matrix into ``(M, Kp*4)`` trits.

    Fully vectorised: ``TRIT_LUT[packed]`` gives ``(M, Kp, 4)`` in one
    NumPy index, then a reshape produces the row-major trit matrix.
    """
    lut = np.asarray(TRIT_LUT, dtype=np.int8)  # (256, 4)
    return lut[packed].reshape(packed.shape[0], packed.shape[1] * 4)


def ternary_matmul_numpy(
    packed: NDArray[np.uint8],
    x: NDArray[np.float16],
    gamma: NDArray[np.float32],
) -> NDArray[np.float16]:
    """One-off ``y = gamma * (W_t @ x)`` -- single batch, CPU-only.

    Parameters
    ----------
    packed : (M, ceil(N/4)) uint8
        Fastpacked weight rows (4 trits / byte).
    x : (N,) float16
        Input activation vector.
    gamma : (M,) float32
        Per-output scaling factor.

    Returns
    -------
    y : (M,) float16
        Output activations.
    """
    N = len(x)
    W = unpack_fastpacked_matrix(packed)[:, :N]  # (M, N)
    y = (W.astype(np.float32) @ x.astype(np.float32)) * gamma.astype(np.float32)
    return y.astype(np.float16)


def ternary_matmul_numpy_batched(
    packed: NDArray[np.uint8],
    x_batch: NDArray[np.float16],
    gamma: NDArray[np.float32],
) -> NDArray[np.float16]:
    """Batched version -- ``(B, N)`` inputs, ``(B, M)`` outputs.

    The weight matrix is decoded once and reused for the whole batch;
    the batched matmul is a single BLAS call.
    """
    B, N = x_batch.shape
    W = unpack_fastpacked_matrix(packed)[:, :N]  # (M, N)
    y = (
        W.astype(np.float32) @ x_batch.astype(np.float32).T
    ).T * gamma.astype(np.float32)  # (B, M)
    return y.astype(np.float16)


__all__ = [
    "decode_fastpacked_row",
    "unpack_fastpacked_matrix",
    "ternary_matmul_numpy",
    "ternary_matmul_numpy_batched",
]
