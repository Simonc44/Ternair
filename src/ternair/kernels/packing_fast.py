"""Fast 2‑bit ternary packing — 4 trits per byte.

Each trit occupies 2 bits::

    00  →  0   (zero)
    01  → +1   (positive)
    10  → -1   (negative)
    11  →  unused / treated as 0

Encoding
--------
* ``trit{-1,0,+1}`` →  ``bits`` where ``bits = (trit & 2) | ((trit + 1) & 1)``.
  Simplified: ``bits = trit + 1`` works for ``{-1, 0, +1} → {0, 1, 2}``,
  then we mask with ``0b11``.
* 4 trits packed little‑endian into one ``uint8``:
  ``byte = bits[0] | (bits[1] << 2) | (bits[2] << 4) | (bits[3] << 6)``

Decoding (bit arithmetic, no table needed)
------------------------------------------
::  

    for pos in range(4):
        bits = (byte >> (2 * pos)) & 0x03
        trit = (bits & 1) - ((bits >> 1) & 1)   # {0, +1, -1}

This avoids branching, modulo / division, and memory lookups — ideal
for Triton / SIMD.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

MODE_FASTPACKED: Literal["fastpacked"] = "fastpacked"


# ---------------------------------------------------------------------------
# Encoding / decoding helpers
# ---------------------------------------------------------------------------

def trit_from_2bit(bits: int) -> int:
    """Convert a 2‑bit trit ``{0,1,2,3}`` to ``{-1,0,+1}``.

    >>> trit_from_2bit(0)
    0
    >>> trit_from_2bit(1)
    1
    >>> trit_from_2bit(2)
    -1
    >>> trit_from_2bit(3)
    0
    """
    return (bits & 1) - ((bits >> 1) & 1)


def bits_from_trit(trit: int) -> int:
    """Encode ``{-1, 0, +1}`` into a 2‑bit value ``{0, 1, 2}``.

    >>> bits_from_trit(-1)
    2
    >>> bits_from_trit(0)
    0
    >>> bits_from_trit(1)
    1
    """
    if trit < 0:
        return 2
    if trit > 0:
        return 1
    return 0


def pack_trits_2bit(trits: np.ndarray) -> np.ndarray:
    """Pack a 1‑D int8 array into ``uint8`` (4 trits per byte).

    Trailing slots (if ``len % 4 != 0``) are filled with 0 (trit 0).

    Returns
    -------
    packed : np.ndarray  shape ``(ceil(N / 4),)`` dtype uint8
    """
    if trits.ndim != 1:
        raise ValueError("pack_trits_2bit expects a 1‑D array")
    N = len(trits)
    pad = (-N) % 4
    if pad:
        trits = np.concatenate([trits, np.zeros(pad, dtype=np.int8)])
    bits = np.where(trits < 0, 2, np.where(trits > 0, 1, 0)).astype(np.uint8)
    packed = (
        bits[0::4]
        | (bits[1::4] << 2)
        | (bits[2::4] << 4)
        | (bits[3::4] << 6)
    )
    return packed.astype(np.uint8)


def unpack_trits_2bit(packed: np.ndarray, length: int) -> np.ndarray:
    """Inverse of :func:`pack_trits_2bit`."""
    if length <= 0:
        return np.zeros(0, dtype=np.int8)
    N = len(packed)
    dst = np.zeros(N * 4, dtype=np.int8)
    for pos in range(4):
        bits = (packed.astype(np.int16) >> (2 * pos)) & 0x03
        dst[pos::4] = (bits & 1) - ((bits >> 1) & 1)
    return dst[:length].astype(np.int8)


# ---------------------------------------------------------------------------
# Pre‑computed lookup table for fast NumPy / C loops
# ---------------------------------------------------------------------------

TRIT_LUT: np.ndarray = np.zeros((256, 4), dtype=np.int8)
for _b in range(256):
    for _p in range(4):
        _bits = (_b >> (2 * _p)) & 0x03
        TRIT_LUT[_b, _p] = trit_from_2bit(_bits)


def lookup_trits(byte: int) -> tuple[int, int, int, int]:
    """Return the four trit values for a packed byte (small & fast)."""
    b = int(byte) & 0xFF
    t0 = TRIT_LUT[b, 0]
    t1 = TRIT_LUT[b, 1]
    t2 = TRIT_LUT[b, 2]
    t3 = TRIT_LUT[b, 3]
    return (t0, t1, t2, t3)


__all__ = [
    "MODE_FASTPACKED",
    "trit_from_2bit",
    "bits_from_trit",
    "pack_trits_2bit",
    "unpack_trits_2bit",
    "TRIT_LUT",
    "lookup_trits",
]
