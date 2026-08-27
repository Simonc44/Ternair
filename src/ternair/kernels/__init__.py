"""Low-level inference kernels for packed ternary matmul.

Two packing formats are supported:

* ``"packed"``  —  5 trits / byte  (base‑3, 1.6 b/v, best density).  
* ``"fastpacked"`` — 4 trits / byte  (2‑bit per trit, 2.0 b/v,
  decode with pure bit ops, **compute‑friendly**).

Submodules
----------
packing_fast
    Fast 2‑bit packing (4 trits/byte) with LUT and bit‑arithmetic decode.
packed_ops
    Pure‑Python / numpy reference for the packed matmul algorithm.
triton_matmul
    Triton (GPU) kernel — operates directly on ``fastpacked`` bytes
    without materialising full FP tensors.
cpu_matmul
    Python wrapper + C++ header with AVX‑512 / ARM NEON intrinsics.
"""

from ternair.kernels.packing_fast import (
    pack_trits_2bit,
    unpack_trits_2bit,
    trit_from_2bit,
    MODE_FASTPACKED,
)
from ternair.kernels.packed_ops import (
    ternary_matmul_numpy,
    ternary_matmul_numpy_batched,
    decode_fastpacked_row,
)
from ternair.kernels.triton_fast import (
    has_triton,
    ternary_matmul_triton,
)
from ternair.kernels.cpu_matmul import (
    has_cpu_backend,
    ternary_matmul_cpp,
)

__all__ = [
    "pack_trits_2bit",
    "unpack_trits_2bit",
    "trit_from_2bit",
    "MODE_FASTPACKED",
    "ternary_matmul_numpy",
    "ternary_matmul_numpy_batched",
    "decode_fastpacked_row",
    "has_triton",
    "ternary_matmul_triton",
    "has_cpu_backend",
    "ternary_matmul_cpp",
]
