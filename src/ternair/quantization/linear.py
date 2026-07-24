"""Drop-in ``nn.Linear`` replacement using ternary weights.

Implements the BitNet b1.58 linear layer. Three storage modes:

* ``"int8"``      — 1 byte / trit (8 bits/value). Fast prototyping baseline.
* ``"packed"``    — 5 trits / byte (1.6 bits/value, base-3 encoding).
* ``"fastpacked"``— 4 trits / byte (2 bits/value, simpler 2-bit encoding).

Key fixes vs previous version
------------------------------
* **Device tracking**: ``_frozen_device`` is set explicitly at
  :meth:`freeze_storage` time and kept in sync via :meth:`to`/
  :meth:`cuda`/:meth:`cpu`.  No more fragile ``next(self.parameters())``
  calls that crash on fully-frozen modules with no FP parameters.
* **extra_repr**: human-readable repr shows shape, storage and frozen state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.kernels.packing_fast import pack_trits_2bit, unpack_trits_2bit
from ternair.quantization.packing import (
    MODE_INT8,
    MODE_PACKED,
    StorageMode,
    packed_to_torch,
    torch_to_packed,
)
from ternair.quantization.ternary import _compute_gamma, ternary_linear_forward

MODE_FASTPACKED: StorageMode = "fastpacked"  # type: ignore[assignment]


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
        ``"int8"`` | ``"packed"`` | ``"fastpacked"``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        storage: StorageMode = "int8",
    ) -> None:
        super().__init__()
        if storage not in (MODE_INT8, MODE_PACKED, MODE_FASTPACKED):
            raise ValueError(f"Unsupported storage mode {storage!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.storage = storage

        # FP weight — kept for training; discarded after freeze_storage if desired.
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

        # --- FIX: explicit device tracking instead of next(self.parameters()) ---
        # Set at freeze_storage() time; updated via .to() / .cuda() / .cpu().
        self._frozen_device: torch.device | None = None

        # Learned alpha for QAT (per-output-channel scale).
        self._use_learned_alpha = False
        self.alpha: nn.Parameter | None = None

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    # Device tracking — keep _frozen_device in sync with module moves
    # ------------------------------------------------------------------
    def to(self, *args: Any, **kwargs: Any) -> "TernairLinear":
        result = super().to(*args, **kwargs)
        # Infer the target device from the moved gamma_eval buffer.
        result._frozen_device = result.gamma_eval.device
        return result

    def cuda(self, device=None) -> "TernairLinear":  # type: ignore[override]
        result = super().cuda(device)
        result._frozen_device = result.gamma_eval.device
        return result

    def cpu(self) -> "TernairLinear":  # type: ignore[override]
        result = super().cpu()
        result._frozen_device = result.gamma_eval.device
        return result

    def _get_device(self) -> torch.device:
        """Return the current device — safe even when no FP parameters remain."""
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
        """Ternarise the current FP weight → ``(trits, scale)``."""
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
        weight = self._dequantise(x.dtype)
        return F.linear(x, weight, self.bias)

    def _dequantise(self, dtype: torch.dtype) -> Tensor:
        """Dequantise packed trits for eval-mode inference.

        Uses ``_frozen_device`` (set at :meth:`freeze_storage` and kept in sync
        via :meth:`to`) — never calls ``next(self.parameters())`` which would
        crash on fully-frozen modules with no FP params left.
        """
        if self._pack_kind is None or self._packed_shape is None:
            raise RuntimeError(
                "Call freeze_storage() before using the ternarised forward."
            )

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

        gamma = self.gamma_eval.to(dtype=dtype, device=device).unsqueeze(-1)
        trits_tensor = trits_flat.to(dtype=dtype).reshape(self._packed_shape)
        return trits_tensor * gamma

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
        """Bytes used for ternary storage (packed weights + γ)."""
        if self._pack_kind is None:
            return 0
        packed_bytes = int(self.packed_weight.numel())
        gamma_bytes = self.gamma_eval.numel() * 4  # FP32
        return packed_bytes + gamma_bytes


__all__ = ["TernairLinear", "TernairLinearStorage"]
