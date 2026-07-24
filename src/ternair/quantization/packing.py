"""Ternary weight packing.

True 1.58-bit storage requires packing ~3 ternary values into 5 bits
because log2(3) ≈ 1.585. We use the common scheme that packs 5 ternary
digits into one ``uint8`` (3^5 = 243 ≤ 256). This yields exactly
1.6 bits/value (8 / 5). For pure 1.58-bit storage the 5-into-8 packing
is already within 1% of the theoretical optimum and is the de-facto
reference implementation used by the community ports of BitNet.

We keep two storage modes in the package to stay transparent:

* :data:`MODE_INT8`  - simplest, ~8 bits/value (1 byte per trit).
* :data:`MODE_PACKED`- ~1.6 bits/value (5 trits per byte).

Both modes store the ternary digits in ``{-1, 0, +1}`` and a single
``float32`` ``γ`` per output row.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

StorageMode = Literal["int8", "packed"]

MODE_INT8: StorageMode = "int8"
MODE_PACKED: StorageMode = "packed"

BITS_PER_VALUE = {MODE_INT8: 8.0, MODE_PACKED: 8.0 / 5.0}


def pack_trits(trits: np.ndarray) -> np.ndarray:
    """Pack a 1-D array of ternary digits ``{-1, 0, +1}`` into uint8.

    Every 5 elements are encoded into one byte using base-3 with the
    mapping ``-1 → 0``, ``0 → 1``, ``+1 → 2``. Output length is
    ``ceil(N / 5)``. Trailing slots are filled with the encoding of
    ``-1`` (i.e. 0) so they decode back to ``-1``, which is harmless
    because we always carry the original length alongside the packed
    buffer.
    """
    if trits.ndim != 1:
        raise ValueError("pack_trits expects a 1-D array")
    if trits.dtype != np.int8:
        trits = trits.astype(np.int8)
    mapping = np.array([0, 1, 2], dtype=np.uint8)
    encoded = mapping[trits + 1]  # -1 → 0, 0 → 1, +1 → 2

    pad = (-encoded.size) % 5
    if pad:
        encoded = np.concatenate([encoded, np.zeros(pad, dtype=np.uint8)])

    encoded = encoded.reshape(-1, 5)
    # base-3 with most-significant digit first; weights = 3**i for i in 0..4.
    weights = (3 ** np.arange(5, dtype=np.uint16)).astype(np.uint16)
    packed = (encoded.astype(np.uint16) * weights[None, :]).sum(axis=1).astype(np.uint8)
    return packed


def unpack_trits(packed: np.ndarray, length: int) -> np.ndarray:
    """Inverse of :func:`pack_trits`, returning a 1-D int8 array."""
    if length <= 0:
        return np.zeros(0, dtype=np.int8)
    digits = np.zeros((packed.size, 5), dtype=np.uint8)
    for i in range(5):
        digits[:, i] = (packed.astype(np.uint16) // (3 ** i)) % 3
    flat = digits.reshape(-1)
    flat = flat[:length]
    return (flat.astype(np.int8) - 1)  # 0 → -1, 1 → 0, 2 → +1


def torch_to_packed(trits: torch.Tensor) -> np.ndarray:
    return pack_trits(trits.detach().cpu().numpy().astype(np.int8).reshape(-1))


def packed_to_torch(packed: np.ndarray, shape: tuple[int, ...], dtype: torch.dtype = torch.int8) -> torch.Tensor:
    flat = unpack_trits(packed, length=int(np.prod(shape)))
    return torch.from_numpy(flat).reshape(shape).to(dtype)


__all__ = [
    "StorageMode",
    "MODE_INT8",
    "MODE_PACKED",
    "BITS_PER_VALUE",
    "pack_trits",
    "unpack_trits",
    "torch_to_packed",
    "packed_to_torch",
]
