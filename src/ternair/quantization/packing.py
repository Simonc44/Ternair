"""Backward-compatibility shim -- re-exports from ``ternair.kernels.packing_base8``.

The canonical implementation lives in
:mod:`ternair.kernels.packing_base8` since v0.6.0.  This module is kept
so that ``from ternair.quantization.packing import ...`` continues to
work for users on v0.5.0 and earlier.

All public symbols are re-exported verbatim.  ``MODE_PACKED`` is
preserved as an alias for ``MODE_BASE8`` so existing ``TernairConfig``
values (``storage="packed"``) keep mapping to the same base-3 coding.
"""

from __future__ import annotations

import warnings

from ternair.kernels.packing_base8 import (  # noqa: F401
    BITS_PER_VALUE,
    MODE_BASE8,
    MODE_INT8,
    MODE_PACKED,
    StorageMode,
    bytes_for,
    pack_trits,
    pack_trits_base8,
    packed_to_torch,
    torch_to_packed,
    unpack_trits,
    unpack_trits_base8,
)

# Heads-up: prefer importing from the new location.
warnings.warn(
    "Importing from 'ternair.quantization.packing' is deprecated since v0.6.0; "
    "use 'ternair.kernels.packing_base8' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BITS_PER_VALUE",
    "MODE_BASE8",
    "MODE_INT8",
    "MODE_PACKED",
    "StorageMode",
    "bytes_for",
    "pack_trits",
    "pack_trits_base8",
    "packed_to_torch",
    "torch_to_packed",
    "unpack_trits",
    "unpack_trits_base8",
]
