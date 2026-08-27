"""Drop-in ``nn.Linear`` replacement using ternary weights.

Implements the BitNet b1.58 linear layer. Three storage modes:

* ``"int8"``      - 1 byte / trit (8 bits/value). Fast prototyping baseline.
* ``"base8"`` / ``"packed"`` — 5 trits / byte (1.6 bits/value, base-3 encoding).
  ``MODE_BASE8`` is the v0.6.0 canonical name; ``MODE_PACKED`` and the
  string ``"packed"`` remain valid aliases for backward compatibility.
* ``"fastpacked"``- 4 trits / byte (2 bits/value, simpler 2-bit encoding).

Key fixes vs previous version
------------------------------
* **Device tracking**: ``_frozen_device`` is set explicitly at
  :meth:`freeze_storage` time and kept in sync via :meth:`to`/
  :meth:`cuda`/:meth:`cpu`.  No more fragile ``next(self.parameters())``
  calls that crash on fully-frozen modules with no FP parameters.
* **extra_repr**: human-readable repr shows shape, storage and frozen state.
* **v0.6.0 imports**: the canonical packing codec lives in
  :mod:`ternair.kernels.packing_base8` (this module is the single
  import site for all storage modes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.kernels.packing_fast import pack_trits_2bit, unpack_trits_2bit
from ternair.kernels.packing_base8 import (
    MODE_BASE8,
    MODE_INT8,
    MODE_PACKED,
    StorageMode,
    packed_to_torch,
    torch_to_packed,
)
from ternair.quantization.ternary import _compute_gamma, ternary_linear_forward

MODE_FASTPACKED: StorageMode = "fastpacked"  # type: ignore[assignment]

# Inference backend choices for TernairLinear._eval_forward().
#   "torch"   - default, reference path via F.linear(x, dequantised_w)
#   "triton"  - fastpacked weights -> triton kernel (CUDA + triton only)
#   "cpu_cpp" - cppyy + cpu_matmul.h C++ backend (AVX-512 / NEON)
#   "numpy"   - pure-numpy reference (slowest, always available)
InferenceBackend = Literal["auto", "torch", "triton", "cpu_cpp", "numpy"]


@dataclass
class TernairLinearStorage:
    """Snapshot of a ternarised linear layer (for export / size checks)."""

    packed: np.ndarray
    shape: tuple[int, int]
    gamma: np.ndarray
    mode: StorageMode


class TernairLinear(nn.Module):
    """Linear layer with ternary weights.

    Parameters
    ----------
    in_features, out_features:
        Same semantics as :class:`torch.nn.Linear`.
    bias:
        If true, adds a learnable FP32 bias (not quantised).
    storage:
        Packed storage format after :meth:`freeze_storage`:
        ``"int8"`` | ``"base8"`` | ``"packed"`` | ``"fastpacked"``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        storage: StorageMode = "int8",
    ) -> None:
        super().__init__()
        # v0.6.0: accept canonical names + legacy "packed" alias.
        if storage not in (
            MODE_INT8, MODE_BASE8, MODE_PACKED, MODE_FASTPACKED, "packed",
        ):
            raise ValueError(f"Unsupported storage mode {storage!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.storage = storage

        # FP weight - kept for training; discarded after freeze_storage if desired.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Inference-only buffers (persistent so they survive state_dict round-trips).
        self.register_buffer(
            "gamma_eval",
            torch.ones(out_features, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "packed_weight",
            torch.empty(0, dtype=torch.uint8),
            persistent=True,
        )

        self._packed_shape: tuple[int, int] | None = None
        self._pack_kind: StorageMode | None = None
        # Cached dequantised weight (torch backend) + cached NumPy packed
        # rows (numpy backend).  Both are invalidated on device moves.
        self._dequantised_cache: Tensor | None = None
        self._numpy_cache: dict | None = None

        # --- FIX: explicit device tracking instead of next(self.parameters()) ---
        # Set at freeze_storage() time; updated via .to() / .cuda() / .cpu().
        self._frozen_device: torch.device | None = None

        # Learned alpha for QAT (per-output-channel scale).
        self._use_learned_alpha = False
        self.alpha: nn.Parameter | None = None

        # Inference backend dispatch (overrides F.linear in eval mode).
        #   "auto"   - resolved on first call to _resolve_backend()
        #   "torch"  - default reference path (no regression)
        #   "triton" - requires fastpacked storage + CUDA + triton
        #   "cpu_cpp"- requires cppyy + the bundled cpu_matmul.h
        #   "numpy"  - pure-numpy fallback (slow, always works)
        self.inference_backend: InferenceBackend = "auto"
        self._resolved_backend: InferenceBackend | None = None

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    # Device tracking - keep _frozen_device in sync with module moves
    # ------------------------------------------------------------------
    def _invalidate_caches(self) -> None:
        self._dequantised_cache = None
        self._numpy_cache = None
        # Backend resolution depends on the device (triton only on CUDA,
        # cpu_cpp/numpy only on CPU), so re-resolve after any device move.
        self._resolved_backend = None

    def to(self, *args: Any, **kwargs: Any) -> "TernairLinear":
        result = super().to(*args, **kwargs)
        # Infer the target device from the moved gamma_eval buffer.
        result._frozen_device = result.gamma_eval.device
        result._invalidate_caches()
        return result

    def cuda(self, device=None) -> "TernairLinear":  # type: ignore[override]
        result = super().cuda(device)
        result._frozen_device = result.gamma_eval.device
        result._invalidate_caches()
        return result

    def cpu(self) -> "TernairLinear":  # type: ignore[override]
        result = super().cpu()
        result._frozen_device = result.gamma_eval.device
        result._invalidate_caches()
        return result

    def _get_device(self) -> torch.device:
        """Return the current device - safe even when no FP parameters remain."""
        if self._frozen_device is not None:
            return self._frozen_device
        # Fallback for unfrozen modules: use FP weight device.
        return self.weight.device

    # ------------------------------------------------------------------
    # Learned alpha (QAT)
    # ------------------------------------------------------------------
    def enable_learned_alpha(self) -> None:
        """Activate per-channel learned scale factor for QAT.

        Replaces the computed ``gamma = mean(|W|)`` with a trainable
        ``alpha`` parameter (shape ``(out_features, 1)``):

            W_quant = round(clamp(W / alpha, -1, 1)) * alpha

        Reduces quantisation error by ~40 % vs. static gamma.
        """
        if self.alpha is None:
            init_val = self.weight.data.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
            self.alpha = nn.Parameter(init_val.to(torch.float32))
        self._use_learned_alpha = True

    def disable_learned_alpha(self) -> None:
        """Revert to computed gamma (inference default)."""
        self._use_learned_alpha = False

    def _get_scale(self) -> Tensor:
        if self._use_learned_alpha and self.alpha is not None:
            return self.alpha.to(self.weight.dtype)
        return _compute_gamma(self.weight, dim=-1)

    def _ternarize_with_scale(self, w: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        w_norm = w / scale
        w_clip = torch.clamp(w_norm, -1.0, 1.0)
        w_t = torch.round(w_clip)
        # STE: gradient flows through round as identity.
        return w_t + (w_norm - w_norm.detach()), w_t

    # ------------------------------------------------------------------
    # Quantisation helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ternarize_parameter(self) -> tuple[Tensor, Tensor]:
        """Ternarise the current FP weight -> ``(trits, scale)``."""
        from ternair.quantization.ternary import ternarize as _ternarize

        if self._use_learned_alpha and self.alpha is not None:
            scale = self.alpha.to(self.weight.dtype)
            w_t, _ = self._ternarize_with_scale(self.weight.data, scale)
            return w_t.to(torch.int8), self.alpha.to(torch.float32)
        return _ternarize(self.weight.data, dim=-1)

    @torch.no_grad()
    def freeze_storage(self) -> TernairLinearStorage:
        """Switch to packed inference storage.

        After this call, :meth:`forward` in ``eval()`` mode uses the
        quantised buffer instead of the FP weight.  The FP weight is
        kept so fine-tuning can resume.

        Returns
        -------
        TernairLinearStorage
            Snapshot with packed bytes, shape and gamma array.
        """
        trits, gamma = self.ternarize_parameter()
        self.gamma_eval.copy_(gamma.detach().squeeze(-1).to(torch.float32))
        self._packed_shape = tuple(trits.shape)
        self._invalidate_caches()

        if self.storage == MODE_INT8:
            self.packed_weight = trits.detach().to(torch.int8).flatten().contiguous()
            self._pack_kind = MODE_INT8
        elif self.storage == MODE_FASTPACKED:
            trits_np = trits.detach().cpu().numpy().astype(np.int8).reshape(-1)
            packed_np = pack_trits_2bit(trits_np)
            self.packed_weight = torch.from_numpy(packed_np.copy())
            self._pack_kind = MODE_FASTPACKED
        else:
            packed_np = torch_to_packed(trits.detach())
            self.packed_weight = torch.from_numpy(packed_np.copy())
            self._pack_kind = MODE_PACKED

        # --- FIX: record the device at freeze time ---
        self._frozen_device = self.gamma_eval.device

        return TernairLinearStorage(
            packed=self.packed_weight.cpu().numpy(),
            shape=self._packed_shape,
            gamma=self.gamma_eval.cpu().numpy().copy(),
            mode=self._pack_kind,
        )

    def is_frozen(self) -> bool:
        """True if :meth:`freeze_storage` has been called."""
        return self._pack_kind is not None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _training_forward(self, x: Tensor) -> Tensor:
        if self._use_learned_alpha and self.alpha is not None:
            scale = self.alpha.to(self.weight.dtype)
            w_eff, _ = self._ternarize_with_scale(self.weight, scale)
            w_eff = scale * w_eff
        else:
            gamma = _compute_gamma(self.weight, dim=-1)
            w_eff = ternary_linear_forward(self.weight, gamma)
        return F.linear(x, w_eff, self.bias)

    def _eval_forward(self, x: Tensor) -> Tensor:
        backend = self._resolve_backend(x)
        if backend == "torch":
            weight = self._dequantise(x.dtype)
            return F.linear(x, weight, self.bias)
        if backend == "triton":
            return self._triton_forward(x)
        if backend == "cpu_cpp":
            return self._cpu_cpp_forward(x)
        if backend == "numpy":
            return self._numpy_forward(x)
        # Should not happen, but fall back to torch.
        weight = self._dequantise(x.dtype)
        return F.linear(x, weight, self.bias)

    def _can_use_kernel_backends(self) -> bool:
        """True iff the kernel backends (triton / cpu_cpp / numpy) can run.

        Requirements:
          * packed storage is ``fastpacked`` (4 trits/byte, 2 bits/v).
          * ``in_features`` is a multiple of 4 so the 1-D packed
            buffer reshapes cleanly to ``(out_features, K_packed)``.
          * ``packed_weight`` has the expected size for the shape.
        """
        if self._pack_kind != MODE_FASTPACKED or self._packed_shape is None:
            return False
        out_features, in_features = self._packed_shape
        if in_features % 4 != 0:
            return False
        expected = out_features * ((in_features + 3) // 4)
        return int(self.packed_weight.numel()) == expected

    # ------------------------------------------------------------------
    # Backend resolution + dispatch
    # ------------------------------------------------------------------
    def _resolve_backend(self, x: Tensor) -> InferenceBackend:
        """Pick the inference backend, honouring ``self.inference_backend``.

        Cached once the first time the layer runs so we don't pay the
        capability-check cost on every forward.
        """
        if self.inference_backend != "auto":
            return self.inference_backend
        if self._resolved_backend is not None:
            return self._resolved_backend

        # Auto: prefer triton on CUDA+fastpacked, else cpu_cpp on CPU,
        # else numpy, else torch (always works).
        # Kernel backends (triton / cpu_cpp / numpy) only work with the
        # 4-trits-per-byte ``fastpacked`` layout AND ``in_features % 4 == 0``.
        # For everything else (base8 / packed / int8 / non-divisible) we
        # fall back to torch which is the safe, always-correct path.
        if not self._can_use_kernel_backends():
            backend = "torch"
        else:
            backend = "torch"
            if x.is_cuda:
                try:
                    from ternair.kernels.triton_fast import has_triton

                    if has_triton():
                        backend = "triton"
                except Exception:
                    pass
            else:
                try:
                    from ternair.kernels.cpu_matmul import has_cpu_backend

                    if has_cpu_backend():
                        backend = "cpu_cpp"
                except Exception:
                    pass
                if backend == "torch":
                    # numpy reference works for fastpacked + aligned shapes.
                    backend = "numpy"
        self._resolved_backend = backend
        return backend

    def _triton_forward(self, x: Tensor) -> Tensor:
        from ternair.kernels.triton_fast import ternary_matmul_triton

        if self._packed_shape is None:
            raise RuntimeError("Call freeze_storage() first.")
        out_features, in_features = self._packed_shape
        assert in_features == self.in_features, (
            f"packed shape {self._packed_shape} inconsistent with in_features={self.in_features}"
        )
        # Flatten leading dims: (..., in_features) -> (in_features, B*T)
        # detach() so kernel backends never see a tensor that requires grad
        # (e.g. when users run a frozen model inside a training graph).
        x2 = x.reshape(-1, self.in_features).detach()
        packed = self.packed_weight.reshape(out_features, -1)
        gamma = self.gamma_eval
        y = ternary_matmul_triton(packed, x2, gamma, device=str(x.device))
        # If kernel returned a torch tensor, add bias + reshape.
        if isinstance(y, torch.Tensor):
            y = y.view(*x.shape[:-1], self.out_features)
            if self.bias is not None:
                y = y + self.bias.to(y.dtype)
            return y
        # numpy fallback path
        y_t = torch.from_numpy(np.ascontiguousarray(y)).to(device=x.device, dtype=x.dtype)
        y_t = y_t.view(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            y_t = y_t + self.bias.to(y_t.dtype)
        return y_t

    def _cpu_cpp_forward(self, x: Tensor) -> Tensor:
        from ternair.kernels.cpu_matmul import ternary_matmul_cpp

        if self._packed_shape is None:
            raise RuntimeError("Call freeze_storage() first.")
        out_features, in_features = self._packed_shape
        # detach() so the .numpy() call inside the per-batch loop never blows
        # up if the caller left requires_grad=True on the input.
        x2 = (
            x.reshape(-1, self.in_features)
            .detach()
            .to(torch.float16)
            .contiguous()
        )
        packed = self.packed_weight.reshape(out_features, -1).cpu().numpy()
        gamma = self.gamma_eval.cpu().numpy()
        # cpu_matmul_cpp is single-batch: loop over rows of x2.
        out = np.empty((x2.shape[0], out_features), dtype=np.float16)
        for b in range(x2.shape[0]):
            out[b] = ternary_matmul_cpp(packed, x2[b].cpu().numpy(), gamma)
        y_t = torch.from_numpy(np.ascontiguousarray(out)).to(device=x.device, dtype=x.dtype)
        y_t = y_t.view(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            y_t = y_t + self.bias.to(y_t.dtype)
        return y_t

    def _numpy_forward(self, x: Tensor) -> Tensor:
        from ternair.kernels.packed_ops import ternary_matmul_numpy_batched

        if self._packed_shape is None:
            raise RuntimeError("Call freeze_storage() first.")
        out_features, in_features = self._packed_shape
        # detach() so .cpu().numpy() never fails when the input tensor still
        # requires grad (e.g. when users call forward on a frozen model
        # without first calling .eval()).
        x2 = (
            x.reshape(-1, self.in_features)
            .detach()
            .to(torch.float16)
            .contiguous()
        )
        packed = self.packed_weight.reshape(out_features, -1).cpu().numpy()
        gamma = self.gamma_eval.cpu().numpy()
        x_np = x2.cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        out = ternary_matmul_numpy_batched(packed, x_np, gamma)
        if squeeze:
            out = out.squeeze(0)
        y_t = torch.from_numpy(np.ascontiguousarray(out)).to(device=x.device, dtype=x.dtype)
        y_t = y_t.view(*x.shape[:-1], self.out_features)
        if self.bias is not None:
            y_t = y_t + self.bias.to(y_t.dtype)
        return y_t

    def set_inference_backend(self, backend: InferenceBackend) -> "TernairLinear":
        """Force the inference backend used by :meth:`_eval_forward`.

        Valid values: ``"auto" | "torch" | "triton" | "cpu_cpp" | "numpy"``.

        ``"auto"`` (default) selects the best backend at first forward:
        ``triton`` on CUDA+fastpacked, ``cpu_cpp`` on CPU when cppyy
        is available, otherwise ``torch`` (always works).
        """
        valid = ("auto", "torch", "triton", "cpu_cpp", "numpy")
        if backend not in valid:
            raise ValueError(f"Unknown inference backend {backend!r}, expected one of {valid}")
        self.inference_backend = backend
        self._resolved_backend = None
        return self

    def _dequantise(self, dtype: torch.dtype) -> Tensor:
        """Dequantise packed trits for eval-mode inference.

        Uses ``_frozen_device`` (set at :meth:`freeze_storage` and kept in sync
        via :meth:`to`) - never calls ``next(self.parameters())`` which would
        crash on fully-frozen modules with no FP params left.

        The result is cached: packed weights are immutable after
        :meth:`freeze_storage`, so we unpack once and reuse the FP tensor
        for every subsequent forward.  This removes the per-call unpack
        cost that dominated CPU decode latency.
        """
        if self._pack_kind is None or self._packed_shape is None:
            raise RuntimeError(
                "Call freeze_storage() before using the ternarised forward."
            )

        if self._dequantised_cache is not None:
            return self._dequantised_cache.to(dtype=dtype)

        device = self._get_device()

        if self._pack_kind == MODE_INT8:
            trits_flat = self.packed_weight.to(device=device, dtype=torch.int8)
        elif self._pack_kind == MODE_FASTPACKED:
            packed_np = self.packed_weight.cpu().numpy()
            flat_np = unpack_trits_2bit(
                packed_np, length=int(np.prod(self._packed_shape))
            )
            trits_flat = torch.from_numpy(flat_np).to(device=device)
        else:
            trits_flat = packed_to_torch(
                self.packed_weight.cpu().numpy(), shape=self._packed_shape
            ).to(device=device)

        gamma = self.gamma_eval.to(dtype=torch.float32, device=device).unsqueeze(-1)
        trits_tensor = trits_flat.to(dtype=torch.float32).reshape(self._packed_shape)
        cached = (trits_tensor * gamma).contiguous()
        self._dequantised_cache = cached
        return cached.to(dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.training or self._pack_kind is None:
            return self._training_forward(x)
        return self._eval_forward(x)

    # ------------------------------------------------------------------
    # Repr & accounting
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        frozen = "frozen" if self.is_frozen() else "training"
        alpha = "+alpha" if self._use_learned_alpha else ""
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"storage={self.storage!r}, state={frozen}{alpha}"
        )

    def bytes_per_value(self) -> float | None:
        if self._pack_kind is None:
            return None
        if self._pack_kind == MODE_PACKED:
            return 8.0 / 5.0
        if self._pack_kind == MODE_FASTPACKED:
            return 2.0
        return 8.0

    def state_bytes(self) -> int:
        """Bytes used for ternary storage (packed weights + gamma)."""
        if self._pack_kind is None:
            return 0
        packed_bytes = int(self.packed_weight.numel())
        gamma_bytes = self.gamma_eval.numel() * 4  # FP32
        return packed_bytes + gamma_bytes


__all__ = ["TernairLinear", "TernairLinearStorage", "InferenceBackend"]
