"""Ternary weight packing -- base-8 (5 trits per byte, 1.6 bits/value).

True 1.58-bit storage requires packing ~3 ternary values into 5 bits
because ``log2(3) = 1.585``. We use the common scheme that packs 5
ternary digits into one ``uint8`` byte (``3**5 = 243 <= 256``). This
yields exactly **1.6 bits/value** (8 / 5), within 1% of the theoretical
1.58 bits/value floor and is the de-facto reference implementation used
by the community ports of BitNet.

Storage modes
-------------
* :data:`MODE_INT8`  - simplest, 8 bits/value (1 byte per trit).
* :data:`MODE_BASE8` - 1.6 bits/value (5 trits per byte, base-3).
* :data:`MODE_FAST`  - 2.0 bits/value (4 trits per byte, fastpacked).
  See :mod:`ternair.kernels.packing_fast`.

Both :data:`MODE_BASE8` and the legacy ``MODE_PACKED`` constant point
to the same encoding so that pre-v0.6.0 user code keeps working
without modification.

Module placement
----------------
This module lives under ``kernels/`` because packing is a
storage-format concern (codec), not a quantisation-policy concern.
The original module at ``ternair.quantization.packing`` is kept as a
thin re-export shim for backward compatibility.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

# Canonical new name.
MODE_BASE8: StorageMode = "base8"  # type: ignore[assignment]

# Re-exported for shim parity: ternair.quantization.packing used to expose it.
MODE_INT8: StorageMode = "int8"  # type: ignore[assignment]

# Backward-compat alias: pre-v0.6.0 users rely on ``MODE_PACKED``.
MODE_PACKED: StorageMode = MODE_BASE8
StorageMode = Literal["int8", "base8", "packed", "fastpacked"]  # type: ignore[assignment]

# Bits-per-value table (also used by size.py and benchmark eval).
BITS_PER_VALUE = {
    "int8": 8.0,
    "base8": 8.0 / 5.0,
    # Legacy alias -- same value as base8.
    "packed": 8.0 / 5.0,
    "fastpacked": 2.0,
}


# ---------------------------------------------------------------------------
# Core pack / unpack
# ---------------------------------------------------------------------------


def pack_trits_base8(trits: np.ndarray) -> np.ndarray:
    """Pack a 1-D array of ternary digits ``{-1, 0, +1}`` into ``uint8``.

    Every 5 elements are encoded into one byte using base-3 with the
    mapping ``-1 -> 0``, ``0 -> 1``, ``+1 -> 2``.  Output length is
    ``ceil(N / 5)``.  Trailing slots are padded with the encoding of
    ``0`` (i.e. 1) so they decode back to ``0``, which is harmless
    because we always carry the original length alongside the
    packed buffer.

    Named ``pack_trits_base8`` internally; the public alias
    :func:`pack_trits` is kept for backward compatibility.
    """
    if trits.ndim != 1:
        raise ValueError("pack_trits_base8 expects a 1-D array")
    if trits.dtype != np.int8:
        trits = trits.astype(np.int8)
    mapping = np.array([0, 1, 2], dtype=np.uint8)
    encoded = mapping[trits + 1]  # -1 -> 0, 0 -> 1, +1 -> 2

    pad = (-encoded.size) % 5
    if pad:
        encoded = np.concatenate([encoded, np.ones(pad, dtype=np.uint8)])

    encoded = encoded.reshape(-1, 5)
    # Least-significant digit first base-3 grouping (weights = 3**i).
    weights = (3 ** np.arange(5, dtype=np.uint16)).astype(np.uint16)
    packed = (encoded.astype(np.uint16) * weights[None, :]).sum(axis=1).astype(np.uint8)
    return packed


def unpack_trits_base8(packed: np.ndarray, length: int) -> np.ndarray:
    """Inverse of :func:`pack_trits_base8` returning a 1-D ``int8`` array."""
    if length <= 0:
        return np.zeros(0, dtype=np.int8)
    digits = np.zeros((packed.size, 5), dtype=np.uint8)
    for i in range(5):
        digits[:, i] = (packed.astype(np.uint16) // (3 ** i)) % 3
    flat = digits.reshape(-1)
    flat = flat[:length]
    return (flat.astype(np.int8) - 1)  # 0 -> -1, 1 -> 0, 2 -> +1


# ---------------------------------------------------------------------------
# Legacy public aliases (kept for backward compatibility).
# ---------------------------------------------------------------------------
pack_trits = pack_trits_base8
unpack_trits = unpack_trits_base8


def torch_to_packed(trits: torch.Tensor) -> np.ndarray:
    """``torch.Tensor`` -> packed ``np.ndarray`` (base-8)."""
    return pack_trits_base8(trits.detach().cpu().numpy().astype(np.int8).reshape(-1))


def packed_to_torch(
    packed: np.ndarray,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """Packed ``np.ndarray`` -> ``torch.Tensor`` with the given shape."""
    flat = unpack_trits_base8(packed, length=int(np.prod(shape)))
    return torch.from_numpy(flat).reshape(shape).to(dtype)


def bytes_for(n_values: int) -> int:
    """Bytes required to store ``n_values`` ternary digits in base-8 mode."""
    return (n_values + 4) // 5


__all__ = [
    "MODE_BASE8",
    "MODE_PACKED",  # alias
    "StorageMode",
    "BITS_PER_VALUE",
    "pack_trits_base8",
    "unpack_trits_base8",
    "pack_trits",  # alias
    "unpack_trits",  # alias
    "torch_to_packed",
    "packed_to_torch",
    "bytes_for",
]
