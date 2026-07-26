"""HuggingFace-compatible ternary model loader.

This module is the **ingestion** counterpart to :mod:`ternair.model.inference`:
it loads a Mistral / LLaMA-style causal LM that has been exported as a
"ternary SafeTensors" bundle, swaps every ``nn.Linear`` for a
:class:`TernaryLinearFast` that unpacks the 2-bit ternary weights on the
fly, and returns a ready-to-``generate()`` model.

Two SafeTensors schemas are auto-detected:

1. **External / LLaMA export** (this module's primary target)

   ``model.layers.0.attention.wq.weight.packed``   -- ``uint8`` packed trits
   ``model.layers.0.attention.wq.weight.alpha``    -- ``float32`` scale
   ``model.layers.0.attention.wq.weight.shape``    -- ``int64`` ``[out, in]``

2. **Native :mod:`ternair.model.export`** (fallback)

   Tensor names like ``...packed_weight`` + ``...gamma_eval``.

Typical use::

    from ternair import load_ternair_model

    model, tokenizer = load_ternair_model(
        r"C:\\Users\\admin\\Documents\\Mistral 7B\\mistral-7b-v0.3-ternair",
        device="cuda",
    )
    inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=True,
                        temperature=0.7, top_p=0.9, repetition_penalty=1.1)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

The :class:`TernaryLinearFast` class is intentionally **separate** from
:class:`ternair.quantization.linear.TernairLinear`: it un-packs the 2-bit
weights on every forward (slower but does not require ``freeze_storage()``)
so it works on any HF model class without coupling to the rest of Ternair.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch import Tensor, nn

_LOGGER = logging.getLogger(__name__)

InferenceBackend = Literal["auto", "torch", "triton", "numpy", "native"]


# ---------------------------------------------------------------------------
# Native C++ engine detection
# ---------------------------------------------------------------------------

def _native_engine_available() -> bool:
    """``True`` iff the native ``libternair_native.so`` is loadable.

    Looks in ``ternair/native/build/`` (g++ build target) and via
    ``ctypes.util.find_library``.  Cached after the first call so we
    don't repeatedly hit the filesystem in the hot ``forward()`` loop.
    """
    cache_attr = "__dict__"  # placeholder, replaced below with module-level cache
    # Module-level memoised cache.
    if "_native_engine_available_cache" not in globals():
        try:
            from ternair.native import available as _native_available
            globals()["_native_engine_available_cache"] = bool(_native_available())
        except Exception:
            globals()["_native_engine_available_cache"] = False
    return globals()["_native_engine_available_cache"]


# ---------------------------------------------------------------------------
# LLaMA-native -> HuggingFace key mapping
# ---------------------------------------------------------------------------

_LLAMA_TO_HF = {
    "attention.wq":   "self_attn.q_proj",
    "attention.wk":   "self_attn.k_proj",
    "attention.wv":   "self_attn.v_proj",
    "attention.wo":   "self_attn.o_proj",
    "feed_forward.w1": "mlp.gate_proj",
    "feed_forward.w2": "mlp.down_proj",
    "feed_forward.w3": "mlp.up_proj",
    "attention_norm":  "input_layernorm",
    "ffn_norm":        "post_attention_layernorm",
}

_ROOT_MAP = {
    "tok_embeddings": "model.embed_tokens",
    "norm":           "model.norm",
    "output":         "lm_head",
}


def llama_to_hf(llama_key: str) -> str | None:
    """Translate a LLaMA-native tensor name to its HuggingFace equivalent.

    Pass-through for keys already in HF form (``model.*`` / ``lm_head``).
    Returns ``None`` when no mapping applies.
    """
    if llama_key.startswith("model.") or llama_key == "lm_head":
        return llama_key

    for src, dst in _ROOT_MAP.items():
        if llama_key == src or llama_key.startswith(src + "."):
            return dst + llama_key[len(src):]

    if llama_key.startswith("layers."):
        parts = llama_key.split(".")
        layer_n = parts[1]
        rest = ".".join(parts[2:])
        for src, dst in _LLAMA_TO_HF.items():
            if rest == src or rest.startswith(src + "."):
                return f"model.layers.{layer_n}.{dst}" + rest[len(src):]
    return None


def _get_mod(root: nn.Module, path: str) -> tuple[nn.Module, str] | None:
    """Walk ``root`` along dotted ``path`` and return ``(parent, attr)``.

    Returns ``None`` if any intermediate attribute is missing.
    """
    parts = path.split(".")
    cur: nn.Module = root
    for p in parts[:-1]:
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur, parts[-1]


# ---------------------------------------------------------------------------
# 2-bit unpack (matches ternair.kernels.packing_fast bit-mapping)
# ---------------------------------------------------------------------------


def unpack_2bit(packed: Tensor, out_f: int, in_f: int) -> Tensor:
    """Decode a packed ``uint8`` buffer (4 trits/byte) into ``(out_f, in_f)``.

    Bit mapping (identical to :func:`ternair.kernels.packing_fast.unpack_trits_2bit`):

    * ``0b00`` -> ``0``
    * ``0b01`` -> ``+1``
    * ``0b10`` -> ``-1``
    * ``0b11`` -> unused (yields ``-1`` -- bit-arithmetic produces ``-1``)

    Parameters
    ----------
    packed
        ``uint8`` tensor of shape ``(out_f * in_f / 4,)`` (or 1-D ``int``).
    out_f, in_f
        Logical output/input sizes.
    """
    total = out_f * in_f
    flat = packed.to(torch.int32).flatten()
    # Stack the 4 trits packed per byte, then flatten to trits-major order.
    trits = torch.stack(
        [
            (flat >> 0) & 3,
            (flat >> 2) & 3,
            (flat >> 4) & 3,
            (flat >> 6) & 3,
        ],
        dim=1,
    ).flatten()
    w = torch.zeros(trits.shape[0], dtype=torch.float32)
    w[trits == 1] = 1.0
    w[trits == 2] = -1.0
    return w[:total].reshape(out_f, in_f)


# ---------------------------------------------------------------------------
# Standalone ternary linear (lazy / on-the-fly unpack)
# ---------------------------------------------------------------------------


class TernaryLinearFast(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that holds 2-bit ternary weights.

    The packed buffer and per-output alpha are stored as non-trainable
    buffers (``register_buffer``) so the module survives ``.to(device)`` /
    ``.half()`` without corrupting the binary payload.

    Three forward paths are dispatched at runtime:

    * ``"triton"`` -- fastpacked weights → ``ternair.kernels.triton_fast``
      GPU kernel (CUDA + triton only).
    * ``"numpy"``  -- fastpacked weights → ``ternair.kernels.packed_ops``
      vectorised NumPy matmul (always available).
    * ``"torch"``  -- pure-Python ``unpack_2bit`` + ``F.linear`` fallback
      (FP32 accumulator; slower but always works; also the path used when
      ``in_f % 4 != 0`` since the packed buffer cannot reshape cleanly).

    The ``backend="auto"`` default picks the best path available on the
    host (triton > numpy > torch).  Use ``set_inference_backend()`` to
    override per-layer.

    Parameters
    ----------
    packed
        ``uint8`` packed trit buffer (``out_f * in_f / 4`` elements).
    alpha
        Per-output scalar (``float32`` or ``float``).
    out_f, in_f
        Logical dimensions.
    bias
        Optional FP bias of shape ``(out_f,)``.
    backend
        Initial inference backend (``"auto" | "torch" | "triton" | "numpy"``).
    """

    def __init__(
        self,
        packed: Tensor,
        alpha: Tensor | float,
        out_f: int,
        in_f: int,
        bias: Optional[Tensor] = None,
        backend: InferenceBackend = "auto",
    ) -> None:
        super().__init__()
        self.out_f = out_f
        self.in_f = in_f
        # CRITICAL: register as buffer (not Parameter) so .half() / .to()
        # leave the binary data intact.
        self.register_buffer("pw", packed.detach().to(torch.uint8))
        self.register_buffer(
            "alpha",
            torch.as_tensor(float(alpha), dtype=torch.float32),
        )
        self.bias_buf: Tensor | None = None
        if bias is not None:
            self.register_buffer("bias_buf", bias.detach().to(torch.float32))

        # Backend dispatch (mirrors TernairLinear.set_inference_backend).
        if backend not in ("auto", "torch", "triton", "numpy", "native"):
            raise ValueError(
                f"Unknown inference backend {backend!r}; expected "
                "one of ('auto', 'torch', 'triton', 'numpy', 'native')"
            )
        self.backend: InferenceBackend = backend
        self._resolved_backend: InferenceBackend | None = None

    # ------------------------------------------------------------------
    # Backend resolution + dispatch
    # ------------------------------------------------------------------
    def set_inference_backend(self, backend: InferenceBackend) -> "TernaryLinearFast":
        """Force the inference backend used by :meth:`forward`.

        Valid values: ``"auto" | "torch" | "triton" | "numpy" | "native"``.
        """
        valid = ("auto", "torch", "triton", "numpy", "native")
        if backend not in valid:
            raise ValueError(f"Unknown inference backend {backend!r}, expected one of {valid}")
        self.backend = backend
        self._resolved_backend = None
        return self

    def _can_use_kernel_backends(self) -> bool:
        """True iff the 1-D packed buffer reshapes cleanly to ``(out_f, in_f/4)``."""
        if self.in_f % 4 != 0:
            _LOGGER.debug(
                "TernaryLinearFast: in_f=%d not divisible by 4, kernel path disabled",
                self.in_f,
            )
            return False
        k_packed = self.in_f // 4
        return int(self.pw.numel()) == self.out_f * k_packed

    def _resolve_backend(self, x: Tensor) -> InferenceBackend:
        requested = self.backend
        if requested != "auto":
            # Even when explicit, ensure the kernel path is actually
            # viable (in_f % 4 == 0, buffer reshape clean, .so present).
            # Otherwise silently fall back to "torch" (with a debug log)
            # instead of crashing on the reshape at first forward.
            if requested in ("triton", "numpy", "native") and not self._can_use_kernel_backends():
                _LOGGER.debug(
                    "TernaryLinearFast: backend=%r requested but constraints "
                    "not met (in_f=%d, out_f=%d, pw.numel()=%d) -> fallback to 'torch'",
                    requested, self.in_f, self.out_f, int(self.pw.numel()),
                )
                self._resolved_backend = "torch"
                return "torch"
            if requested == "native" and not _native_engine_available():
                _LOGGER.debug(
                    "TernaryLinearFast: backend='native' requested but "
                    "libternair_native.so not found -> fallback to 'numpy'",
                )
                self._resolved_backend = "numpy"
                return "numpy"
            return requested
        if self._resolved_backend is not None:
            return self._resolved_backend
        if not self._can_use_kernel_backends():
            self._resolved_backend = "torch"
            return "torch"
        backend: InferenceBackend = "torch"
        if x.is_cuda:
            try:
                from ternair.kernels.triton_fast import has_triton

                if has_triton():
                    backend = "triton"
            except Exception:
                pass
        else:
            # CPU: prefer the C++ native engine (AVX-512 / AVX-2).
            # Fall back to numpy -> torch in that order.
            if _native_engine_available():
                backend = "native"
            else:
                try:
                    import numpy  # noqa: F401
                    backend = "numpy"
                except Exception:
                    pass
        self._resolved_backend = backend
        return backend

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        backend = self._resolve_backend(x)
        if backend == "triton":
            return self._triton_forward(x)
        if backend == "native":
            return self._native_forward(x)
        if backend == "numpy":
            return self._numpy_forward(x)
        return self._torch_forward(x)

    def _torch_forward(self, x: Tensor) -> Tensor:
        """Pure-Python fallback (FP32 accumulator). Always works."""
        w = (
            unpack_2bit(self.pw, self.out_f, self.in_f).to(x.device)
            * self.alpha.to(x.device)
        )
        out = torch.nn.functional.linear(x.float(), w)
        if self.bias_buf is not None:
            out = out + self.bias_buf.to(x.device)
        return out.to(x.dtype)

    def _triton_forward(self, x: Tensor) -> Tensor:
        """``triton_fast`` kernel path (CUDA + fastpacked)."""
        from ternair.kernels.triton_fast import ternary_matmul_triton

        k_packed = self.in_f // 4
        packed_2d = self.pw.view(self.out_f, k_packed)
        x_flat = x.reshape(-1, self.in_f)
        # alpha is stored as a 0-d scalar; the kernel expects (M,) -- broadcast.
        gamma = self.alpha.repeat(self.out_f)
        y = ternary_matmul_triton(
            packed_2d, x_flat, gamma, device=str(x.device)
        )
        if isinstance(y, torch.Tensor):
            y = y.view(*x.shape[:-1], self.out_f)
            if self.bias_buf is not None:
                y = y + self.bias_buf.to(device=y.device, dtype=y.dtype)
            return y.to(x.dtype)
        # Kernel fell back to numpy internally.
        y_t = torch.from_numpy(np.ascontiguousarray(y)).to(
            device=x.device, dtype=x.dtype
        )
        y_t = y_t.view(*x.shape[:-1], self.out_f)
        if self.bias_buf is not None:
            y_t = y_t + self.bias_buf.to(device=y_t.device, dtype=y_t.dtype)
        return y_t

    def _numpy_forward(self, x: Tensor) -> Tensor:
        """``packed_ops`` NumPy path (vectorised; FP16 accumulator)."""
        from ternair.kernels.packed_ops import ternary_matmul_numpy_batched

        k_packed = self.in_f // 4
        packed_2d = self.pw.view(self.out_f, k_packed).cpu().numpy()
        # alpha is 0-d; broadcast to (M,) for the kernel.
        gamma_np = self.alpha.repeat(self.out_f).cpu().numpy()
        x_flat = x.reshape(-1, self.in_f).to(torch.float16).cpu().numpy()
        squeeze = x_flat.ndim == 1
        if squeeze:
            x_flat = x_flat[np.newaxis, :]
        y_np = ternary_matmul_numpy_batched(packed_2d, x_flat, gamma_np)
        if squeeze:
            y_np = y_np.squeeze(0)
        y_t = torch.from_numpy(np.ascontiguousarray(y_np)).to(
            device=x.device, dtype=x.dtype
        )
        y_t = y_t.view(*x.shape[:-1], self.out_f)
        if self.bias_buf is not None:
            y_t = y_t + self.bias_buf.to(device=y_t.device, dtype=y_t.dtype)
        return y_t

    def _native_forward(self, x: Tensor) -> Tensor:
        """Host C++ engine via ctypes (``ternair.native.native.ternary_matmul``).

        Backend dispatch at runtime picks AVX-512 / AVX-2 / scalar based
        on CPU feature detection.  CPU only -- CUDA inputs trigger a
        host round-trip and lose perf (this path is therefore never
        selected via ``backend="auto"`` on CUDA).

        For batch dimension B > 1 we iterate in Python (autoregressive
        generation uses B == 1 -- zero overhead).  Allocations are
        per-call (~64 KB on a typical model); caching buffers here would
        create thread-safety issues for higher batch sizes.
        """
        from ternair.native import ternary_matmul

        k_packed = self.in_f // 4
        # Pack inputs once; per-batch row dispatch follows.
        packed_2d = self.pw.view(self.out_f, k_packed).cpu().numpy()
        gamma_1d = self.alpha.repeat(self.out_f).cpu().numpy()
        x_flat = x.reshape(-1, self.in_f)
        B = x_flat.shape[0]

        y_rows: list[Tensor] = []
        for i in range(B):
            # Cast to fp16 and view as raw uint16 bits (matches the C ABI).
            x_fp16 = x_flat[i].to(torch.float16).cpu().numpy()
            x_bits = x_fp16.view(np.uint16)
            y_bits = ternary_matmul(packed_2d, x_bits, gamma_1d)  # (out_f,) uint16
            y_fp16 = y_bits.view(np.float16)
            y_rows.append(torch.from_numpy(np.ascontiguousarray(y_fp16)))

        out = torch.stack(y_rows, dim=0).view(*x.shape[:-1], self.out_f)
        out = out.to(device=x.device, dtype=x.dtype)
        if self.bias_buf is not None:
            out = out + self.bias_buf.to(device=out.device, dtype=out.dtype)
        return out

    def extra_repr(self) -> str:
        backend = self._resolved_backend or self.backend
        return (
            f"in={self.in_f}, out={self.out_f}, alpha={self.alpha.item():.4f}, "
            f"backend={backend!r}"
        )


# ---------------------------------------------------------------------------
# Loader result
# ---------------------------------------------------------------------------


@dataclass
class LoadReport:
    """Diagnostic summary returned by :func:`load_ternair_model`."""

    n_ternary_layers: int = 0
    n_fp16_tensors: int = 0
    n_meta_materialised: int = 0
    n_ignored: int = 0
    schema: str = "external"  # "external" (LLaMA export) | "native" (ternair.export)
    n_skipped_duplicates: int = 0
    ignored_keys: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "n_ternary_layers": self.n_ternary_layers,
            "n_fp16_tensors": self.n_fp16_tensors,
            "n_meta_materialised": self.n_meta_materialised,
            "n_ignored": self.n_ignored,
            "n_skipped_duplicates": self.n_skipped_duplicates,
            "ignored_keys": self.ignored_keys[:10],
        }


# ---------------------------------------------------------------------------
# Loader core
# ---------------------------------------------------------------------------


def _safe_open(path: str):
    """Lazily import safetensors + return the safe_open context manager."""
    from safetensors import safe_open  # type: ignore

    return safe_open(path, framework="pt", device="cpu")


def _auto_build_model(config, model_class=None) -> nn.Module:
    """Build the HF model on the meta device (no RAM allocated yet)."""
    from transformers import AutoModelForCausalLM  # type: ignore

    if model_class is None:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    else:
        with torch.device("meta"):
            model = model_class(config)
    return model


def _load_tokenizer(model_dir: str):
    """Load the tokenizer from ``model_dir``.

    Returns ``None`` if no tokenizer can be loaded (e.g. test fixtures).
    """
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:  # pragma: no cover - depends on optional tokenizer
        _LOGGER.info("Tokenizer not loaded (%s) -- returning None", exc)
        return None


def _detect_schema(tensors: dict[str, Tensor]) -> str:
    """Return ``"external"`` if any ``.weight.packed`` key is present, else ``"native"``."""
    for k in tensors:
        if k.endswith(".weight.packed"):
            return "external"
    return "native"


def _materialise_meta(model: nn.Module) -> int:
    """Replace every meta Parameter / Buffer in ``model`` by a zero tensor on CPU."""
    n = 0
    for mod in model.modules():
        for pn, p in list(mod.named_parameters(recurse=False)):
            if p is not None and p.is_meta:
                setattr(
                    mod,
                    pn,
                    nn.Parameter(torch.zeros(p.shape, dtype=p.dtype), requires_grad=False),
                )
                n += 1
        for bn, b in list(mod.named_buffers(recurse=False)):
            if b is not None and b.is_meta:
                setattr(mod, bn, torch.zeros(b.shape, dtype=b.dtype))
                n += 1
    return n


def load_ternair_model(
    model_dir: str,
    device: str | None = None,
    safetensors_name: str = "model_ternair_2bit.safetensors",
    model_class=None,
    dtype: torch.dtype = torch.float16,
    backend: InferenceBackend = "auto",
) -> tuple[nn.Module, Any, LoadReport]:
    """Load a Mistral / LLaMA-architecture ternary model in one call.

    Parameters
    ----------
    model_dir
        Directory holding ``config.json``, the tokenizer files, and
        ``safetensors_name`` (default ``model_ternair_2bit.safetensors``).
    device
        Target device (``"cuda"`` / ``"cpu"``). ``None`` = auto-detect.
    safetensors_name
        Filename of the ternary bundle inside ``model_dir``.
    model_class
        Optional override of the model class (default: ``AutoModelForCausalLM``).
    dtype
        Target float dtype for FP16 weights (default ``float16``).

    Returns
    -------
    (model, tokenizer, report)
        ``model`` is in ``eval()`` mode on the target device and ready
        for ``model.generate(...)``.  ``tokenizer`` may be ``None`` if
        none was found in ``model_dir``.  ``report`` summarises what was
        loaded.
    """
    from transformers import AutoConfig  # type: ignore

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    report = LoadReport()
    safetensors_path = os.path.join(model_dir, safetensors_name)
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Missing ternary safetensors at {safetensors_path}")

    # 1) Tokenizer + config
    tokenizer = _load_tokenizer(model_dir)
    config = AutoConfig.from_pretrained(model_dir)

    # 2) Build model on meta device
    model = _auto_build_model(config, model_class=model_class)

    # 3) Load safetensors into RAM
    tensors: dict[str, Tensor] = {}
    with _safe_open(safetensors_path) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    report.schema = _detect_schema(tensors)

    # 4) Walk the schema and replace layers
    replaced_modules: set[str] = set()
    if report.schema == "external":
        _replace_external(model, tensors, replaced_modules, report, backend)
    else:
        _replace_native(model, tensors, replaced_modules, report, backend)

    # 5) Load FP16 params (norms, embed, lm_head, biases) for keys not yet handled
    _load_fp16_remaining(model, tensors, replaced_modules, dtype, report)

    # 6) Free RAM
    del tensors
    gc.collect()

    # 7) Materialise any leftover meta tensors (norms with zero weight, etc.)
    report.n_meta_materialised = _materialise_meta(model)

    # 8) Move to device + eval()
    model.to(device)
    model.eval()
    return model, tokenizer, report


# ---------------------------------------------------------------------------
# Schema-specific replacers
# ---------------------------------------------------------------------------


def _replace_external(
    model: nn.Module,
    tensors: dict[str, Tensor],
    replaced: set[str],
    report: LoadReport,
    backend: InferenceBackend = "auto",
) -> None:
    """Replace HF layers using the LLaMA ``.weight.packed / .alpha / .shape`` schema."""
    for key in list(tensors.keys()):
        if not key.endswith(".weight.packed"):
            continue
        base = key[: -len(".packed")]
        alpha_key = base + ".alpha"
        shape_key = base + ".shape"
        if alpha_key not in tensors or shape_key not in tensors:
            continue
        shape = tuple(tensors[shape_key].tolist())
        if len(shape) != 2:
            continue
        out_f, in_f = int(shape[0]), int(shape[1])

        llama_mod = base[: -len(".weight")]
        hf_mod = llama_to_hf(llama_mod)
        if hf_mod is None or hf_mod in replaced:
            report.n_skipped_duplicates += 1
            continue

        res = _get_mod(model, hf_mod)
        if res is None:
            report.n_ignored += 1
            if len(report.ignored_keys) < 10:
                report.ignored_keys.append(hf_mod)
            continue

        parent, attr = res
        bias_t = tensors.get(llama_mod + ".bias", tensors.get(hf_mod + ".bias"))

        if hf_mod == "model.embed_tokens":
            # Embedding holds integer token IDs, not a Linear matmul.
            # Permanently decode to FP16 (negligible extra RAM).
            w = unpack_2bit(tensors[key], out_f, in_f) * float(tensors[alpha_key].item())
            embed_mod = getattr(parent, attr)
            embed_mod.weight = nn.Parameter(
                w.to(dtype=torch.float16), requires_grad=False
            )
        else:
            setattr(
                parent,
                attr,
                TernaryLinearFast(
                    tensors[key],
                    float(tensors[alpha_key].item()),
                    out_f,
                    in_f,
                    bias_t,
                    backend=backend,
                ),
            )
        replaced.add(hf_mod)
        report.n_ternary_layers += 1


def _replace_native(
    model: nn.Module,
    tensors: dict[str, Tensor],
    replaced: set[str],
    report: LoadReport,
    backend: InferenceBackend = "auto",
) -> None:
    """Fallback path: standard ``packed_weight`` / ``gamma_eval`` keys.

    Assumes the model was saved with :func:`ternair.model.export.export_to_safetensors`
    and the per-tensor naming ``<prefix>.packed_weight`` / ``<prefix>.gamma_eval``.
    """
    for key, tensor in tensors.items():
        if not key.endswith(".packed_weight"):
            continue
        prefix = key[: -len(".packed_weight")]
        gamma_key = prefix + ".gamma_eval"
        shape_key = prefix + ".shape"
        if gamma_key not in tensors:
            continue
        # shape is optional -- use the gamma shape when missing.
        if shape_key in tensors:
            shape = tuple(tensors[shape_key].tolist())
            if len(shape) == 2:
                out_f, in_f = int(shape[0]), int(shape[1])
            else:
                out_f, in_f = int(tensor.shape[0]), int(tensors[gamma_key].shape[0])
        else:
            out_f = int(tensors[gamma_key].shape[0])
            packed_len = int(tensor.shape[0])
            in_f = packed_len * 4 // max(out_f, 1)

        # Convert the prefix into an HF dotted path: model.layers.X....
        hf_mod = _prefix_to_hf_path(prefix)
        if hf_mod is None or hf_mod in replaced:
            report.n_skipped_duplicates += 1
            continue
        res = _get_mod(model, hf_mod)
        if res is None:
            report.n_ignored += 1
            continue
        parent, attr = res
        setattr(
            parent,
            attr,
            TernaryLinearFast(
                tensor, tensors[gamma_key], out_f, in_f, backend=backend
            ),
        )
        replaced.add(hf_mod)
        report.n_ternary_layers += 1


def _prefix_to_hf_path(prefix: str) -> str | None:
    """Best-effort conversion of a saved-tensor prefix to an HF dotted path.

    For native ternair exports the prefix is typically already an HF path
    like ``model.layers.0.self_attn.q_proj`` -- we just return it as-is.
    """
    if prefix.startswith("model.") or prefix == "lm_head":
        return prefix
    return None


def _load_fp16_remaining(
    model: nn.Module,
    tensors: dict[str, Tensor],
    replaced: set[str],
    dtype: torch.dtype,
    report: LoadReport,
) -> None:
    """Load FP16 weights / biases for keys not handled by the ternary pass."""
    skip_suffixes = (".packed", ".alpha", ".shape", ".packed_weight", ".gamma_eval")
    for k, tensor in tensors.items():
        if any(k.endswith(s) for s in skip_suffixes):
            continue
        hf_k = llama_to_hf(k)
        if hf_k is None:
            report.n_ignored += 1
            continue
        mod_name = hf_k[: -len(".weight")] if hf_k.endswith(".weight") else hf_k
        if mod_name in replaced:
            continue
        res = _get_mod(model, hf_k)
        if res is None:
            continue
        parent, attr = res
        val = tensor.to(dtype)
        existing = getattr(parent, attr, None)
        if isinstance(existing, nn.Parameter):
            setattr(parent, attr, nn.Parameter(val, requires_grad=False))
        else:
            setattr(parent, attr, val)
        report.n_fp16_tensors += 1


__all__ = [
    "TernaryLinearFast",
    "InferenceBackend",
    "LoadReport",
    "load_ternair_model",
    "llama_to_hf",
    "unpack_2bit",
]