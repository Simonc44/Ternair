"""Python ctypes wrapper around the Ternair native C++ engine.

This module auto-loads ``libternair_native.so`` (or whatever path the
build produced) and exposes a high-level :class:`NativeEngine` that
mirrors the C API but speaks Python.

Typical use::

    from ternair.native.native import NativeEngine, available

    if not available():
        raise RuntimeError("libternair_native.so not built -- run scripts/build.sh")

    eng = NativeEngine("/path/to/model.safetensors", num_threads=0)
    print("backend:", eng.backend_name, "vocab:", eng.vocab_size)

    out = eng.generate([1, 544, 12], max_new_tokens=16,
                       temperature=0.7, top_k=40, top_p=0.9)
    print(out)

The wrapper is intentionally minimal: it passes numpy arrays / Python
lists through ctypes and returns primitives.  For real tokenization
production code, use the loader module + a SentencePiece or AutoTokenizer.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER, c_int, c_int32, c_uint8, c_uint16, c_float, c_char_p, c_void_p,
)
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _find_library() -> Optional[str]:
    """Locate libternair_native.so in common build locations + LD_LIBRARY_PATH."""
    env_path = os.environ.get("TERNAIR_NATIVE_LIB")
    if env_path and os.path.exists(env_path):
        return env_path
    # Walk the package tree.
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "build", "libternair_native.so"),
        os.path.join(here, "build", "ternair_native.so"),
        os.path.join(here, "build", "Release", "ternair_native.dll"),
        os.path.join(here, "build", "Debug", "ternair_native.dll"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # System path
    for name in ("ternair_native", "libternair_native.so",
                 "libternair_native.dylib", "ternair_native.dll"):
        try:
            return ctypes.util.find_library(name)  # type: ignore
        except Exception:
            continue
    return None


def available() -> bool:
    """Return ``True`` iff the native shared library is loadable."""
    return _find_library() is not None


# ---------------------------------------------------------------------------
# Library binding
# ---------------------------------------------------------------------------

class _Lib:
    def __init__(self, path: str) -> None:
        self._lib = ctypes.CDLL(path)
        # ternair_create -> TernairRuntime*
        self._lib.ternair_create.restype = c_void_p
        self._lib.ternair_create.argtypes = []
        # ternair_free
        self._lib.ternair_free.restype = None
        self._lib.ternair_free.argtypes = [c_void_p]
        # ternair_load
        self._lib.ternair_load.restype = c_int
        self._lib.ternair_load.argtypes = [c_void_p, c_char_p, c_int]
        # ternair_backend
        self._lib.ternair_backend.restype = c_int
        self._lib.ternair_backend.argtypes = [c_void_p]
        # ternair_backend_name
        self._lib.ternair_backend_name.restype = c_char_p
        self._lib.ternair_backend_name.argtypes = [c_int]
        # int getters
        for name in (
            "ternair_num_layers",
            "ternair_hidden_size",
            "ternair_intermediate_size",
            "ternair_num_attention_heads",
            "ternair_num_kv_heads",
            "ternair_vocab_size",
            "ternair_max_seq_len",
        ):
            fn = getattr(self._lib, name)
            fn.restype = c_int
            fn.argtypes = [c_void_p]
        # ternair_forward
        self._lib.ternair_forward.restype = c_int
        self._lib.ternair_forward.argtypes = [
            c_void_p, POINTER(c_int32), c_int, POINTER(c_float)
        ]
        # ternair_generate
        self._lib.ternair_generate.restype = c_int
        self._lib.ternair_generate.argtypes = [
            c_void_p, POINTER(c_int32), c_int, POINTER(c_int32), c_int,
            c_int, c_float, c_int, c_float, c_float
        ]
        # ternair_ternary_matmul (raw benchmark / test)
        self._lib.ternair_ternary_matmul.restype = c_int
        self._lib.ternair_ternary_matmul.argtypes = [
            POINTER(c_uint8), c_int, c_int,
            POINTER(c_uint16), c_int,
            POINTER(c_float),
            POINTER(c_uint16)
        ]

    @property
    def raw(self) -> ctypes.CDLL:
        return self._lib


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class NativeEngine:
    """Thin Python wrapper over the Ternair native C++ runtime."""

    def __init__(self, model_path: str, num_threads: int = 0) -> None:
        lib_path = _find_library()
        if lib_path is None:
            raise RuntimeError(
                "libternair_native.so not found. Build with: "
                "bash src/ternair/native/scripts/build.sh"
            )
        self._lib = _Lib(lib_path)
        self._handle = self._lib.raw.ternair_create()
        if not self._handle:
            raise RuntimeError("ternair_create returned NULL")
        c_path = model_path.encode("utf-8")
        rc = self._lib.raw.ternair_load(self._handle, c_path, num_threads)
        if rc != 0:
            self._lib.raw.ternair_free(self._handle)
            self._handle = None
            raise RuntimeError(f"ternair_load({model_path!r}) failed: rc={rc}")
        self.model_path = model_path
        # Cache high-level metadata.
        self._vocab = self._lib.raw.ternair_vocab_size(self._handle)
        self._hidden = self._lib.raw.ternair_hidden_size(self._handle)
        self._num_layers = self._lib.raw.ternair_num_layers(self._handle)
        self._backend_int = self._lib.raw.ternair_backend(self._handle)
        bname = self._lib.raw.ternair_backend_name(self._backend_int)
        self._backend_name = bname.decode("utf-8") if isinstance(bname, bytes) else bname

    @property
    def backend(self) -> int:
        """Numeric backend tag (0=scalar, 1=avx2, 2=avx512)."""
        return self._backend_int

    @property
    def backend_name(self) -> str:
        """Human-readable backend name."""
        return self._backend_name

    @property
    def vocab_size(self) -> int: return self._vocab
    @property
    def hidden_size(self) -> int: return self._hidden
    @property
    def num_layers(self) -> int: return self._num_layers

    def forward(self, input_ids: Sequence[int]) -> list[float]:
        """Compute logits for the last token of ``input_ids``."""
        if not self._handle:
            raise RuntimeError("engine not initialised")
        ids = (c_int32 * len(input_ids))(*input_ids)
        logits = (c_float * self._vocab)()
        rc = self._lib.raw.ternair_forward(
            self._handle, ids, len(input_ids), logits)
        if rc != 0:
            raise RuntimeError(f"ternair_forward failed: rc={rc}")
        return [logits[i] for i in range(self._vocab)]

    def generate(
        self,
        prompt: Sequence[int],
        max_new_tokens: int = 16,
        eos_token_id: int = -1,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> list[int]:
        """Run the generate() loop and return the full token list (prompt + new)."""
        if not self._handle:
            raise RuntimeError("engine not initialised")
        p = (c_int32 * len(prompt))(*prompt)
        out_len = len(prompt) + max_new_tokens
        out = (c_int32 * out_len)()
        n = self._lib.raw.ternair_generate(
            self._handle, p, len(prompt), out, max_new_tokens,
            eos_token_id, c_float(temperature), top_k, c_float(top_p),
            c_float(repetition_penalty))
        if n < 0:
            raise RuntimeError(f"ternair_generate failed: rc={n}")
        return [out[i] for i in range(n)]

    def close(self) -> None:
        if self._handle:
            self._lib.raw.ternair_free(self._handle)
            self._handle = None

    def __enter__(self) -> "NativeEngine":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        try: self.close()
        except Exception: pass


# ---------------------------------------------------------------------------
# Standalone benchmark for the matmul
# ---------------------------------------------------------------------------

def ternary_matmul(
    packed: "np.ndarray",  # (M, Kp) uint8
    x_fp16: "np.ndarray",  # (N,)    fp16 raw bits
    gamma:  "np.ndarray",  # (M,)    fp32
) -> "np.ndarray":
    """Call the raw ternary matmul from Python (for benchmarking)."""
    import numpy as np
    lib = _Lib(_find_library() or "")
    M, Kp = packed.shape
    N = x_fp16.shape[0]
    out = np.zeros(M, dtype=np.uint16)
    rc = lib.raw.ternair_ternary_matmul(
        packed.ctypes.data_as(POINTER(c_uint8)),
        c_int(M), c_int(Kp),
        x_fp16.ctypes.data_as(POINTER(c_uint16)),
        c_int(N),
        gamma.ctypes.data_as(POINTER(c_float)),
        out.ctypes.data_as(POINTER(c_uint16)),
    )
    if rc != 0:
        raise RuntimeError(f"ternair_ternary_matmul failed: rc={rc}")
    return out


def benchmark(M: int = 256, N: int = 256, batch: int = 1, num_runs: int = 20) -> dict:
    """Quick benchmark: compare native vs numpy for a ternary matmul."""
    import time
    import numpy as np
    from ternair.kernels.packing_fast import pack_trits_2bit
    from ternair.kernels.packed_ops import ternary_matmul_numpy_batched

    trits = np.random.randint(-1, 2, size=(M, N)).astype(np.int8).reshape(-1)
    packed = pack_trits_2bit(trits).reshape(M, (N + 3) // 4).astype(np.uint8)
    gamma = np.random.rand(M).astype(np.float32)
    x = np.random.randn(batch, N).astype(np.float16)
    x_view = x.view(np.uint16).reshape(-1)

    # Warmup
    for _ in range(3):
        ternary_matmul(packed, x_view, gamma)
        ternary_matmul_numpy_batched(packed, x, gamma)

    np_times, nt_times = [], []
    for _ in range(num_runs):
        s = time.perf_counter()
        ternary_matmul_numpy_batched(packed, x, gamma)
        np_times.append(time.perf_counter() - s)

        s = time.perf_counter()
        ternary_matmul(packed, x_view, gamma)
        nt_times.append(time.perf_counter() - s)

    return {
        "M": M, "N": N, "batch": batch,
        "numpy_mean_ms": float(np.mean(np_times) * 1000),
        "native_mean_ms": float(np.mean(nt_times) * 1000),
        "speedup": float(np.mean(np_times) / max(np.mean(nt_times), 1e-9)),
    }


__all__ = ["NativeEngine", "available", "ternary_matmul", "benchmark"]
