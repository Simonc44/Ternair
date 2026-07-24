"""Drop-in ``nn.Linear`` replacement using ternary weights.

Implements the BitNet b1.58 linear layer. Two storage modes are
supported:

* ``"int8"``  - one byte per ternary digit (8 bits/value, the
  conservative baseline). Useful for fast prototyping on devices that
  do not benefit from tight packing.
* ``"packed"``- 5 ternary digits into one ``uint8`` byte
  (1.6 bits/value, ~98.5% of the theoretical 1.58 bits/value floor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.kernels.packing_fast import pack_trits_2bit, unpack_trits_2bit
from ternair.kernels.packing_base8 import (
    MODE_BASE8,
    MODE_PACKED,
    packed_to_torch,
    torch_to_packed,
)
from ternair.quantization.ternary import _compute_gamma, ternary_linear_forward

# v0.6.0: canonical storage names are "int8", "base8", "fastpacked".
# "packed" remains as a legacy alias for "base8".
StorageMode = Literal["int8", "base8", "packed", "fastpacked"]
MODE_INT8: StorageMode = "int8"
MODE_FASTPACKED: StorageMode = "fastpacked"


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
        If true, adds an FP32 bias (not quantised, kept for numerical
        stability of the demo).
    storage:
        How to materialise weights after :meth:`freeze_storage`:

        * ``"int8"``: one int8 weight per element (8 bits/value).
        * ``"packed"``: pack 5 trits per byte (1.6 bits/value).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        storage: StorageMode = "int8",
    ) -> None:
        super().__init__()
        # v0.6.0: accept canonical names + the legacy "packed" alias.
        if storage not in (MODE_INT8, MODE_BASE8, MODE_PACKED, MODE_FASTPACKED, "packed"):
            raise ValueError(f"Unsupported storage mode {storage!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.storage = storage

        # FP weight kept only to support continued training; after
        # ``freeze_storage`` the quantised buffer is used instead.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Inference-only buffers
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

        # Alpha appris pour QAT (facteur d'echelle entrainable par canal)
        # Si active, remplace le gamma calcule par un parametre appris.
        self._use_learned_alpha = False
        self.alpha: nn.Parameter | None = None

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    # Alpha appris (QAT) - facteur d'echelle entrainable par canal
    # ------------------------------------------------------------------
    def enable_learned_alpha(self) -> None:
        """Active l'alpha appris pour la distillation QAT.

        Le facteur d'echelle alpha devient un parametre entrainable
        par canal (out_features, 1). Au lieu de calculer gamma = mean(|W|),
        on utilise alpha comme facteur d'echelle :
            W_quant = round(clamp(W / alpha, -1, 1)) * alpha
        """
        if self.alpha is None:
            # Initialiser alpha avec la moyenne des poids (comme gamma)
            init_val = self.weight.data.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
            self.alpha = nn.Parameter(init_val.to(torch.float32))
        self._use_learned_alpha = True

    def disable_learned_alpha(self) -> None:
        """Desactive l'alpha appris et revient au gamma calcule."""
        self._use_learned_alpha = False

    def _get_scale(self) -> Tensor:
        """Retourne le facteur d'echelle (alpha appris ou gamma calcule)."""
        if self._use_learned_alpha and self.alpha is not None:
            return self.alpha.to(self.weight.dtype)
        return _compute_gamma(self.weight, dim=-1)

    def _ternarize_with_scale(self, w: Tensor, scale: Tensor) -> Tensor:
        """Ternarise avec un facteur d'echelle et STE."""
        w_norm = w / scale
        w_clip = torch.clamp(w_norm, -1.0, 1.0)
        w_t = torch.round(w_clip)
        # STE : gradient traverse round comme identite
        return w_t + (w_norm - w_norm.detach()), w_t

    # ------------------------------------------------------------------
    # Quantisation helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ternarize_parameter(self) -> tuple[Tensor, Tensor]:
        """Ternarise the current FP weight, returning ``(trits, gamma)``."""
        from ternair.quantization.ternary import ternarize as _ternarize

        if self._use_learned_alpha and self.alpha is not None:
            scale = self.alpha.to(self.weight.dtype)
            w_t, _ = self._ternarize_with_scale(self.weight.data, scale)
            return w_t.to(torch.int8), self.alpha.to(torch.float32)
        return _ternarize(self.weight.data, dim=-1)

    @torch.no_grad()
    def freeze_storage(self) -> TernairLinearStorage:
        """Switch the layer to inference storage backed by packed trits.

        Subsequent forward passes (while ``self.training is False``)
        will use the quantised buffer instead of the FP weight tensor.
        The original FP ``weight`` parameter is kept around so that
        fine-tuning can resume if needed.
        """
        trits, gamma = self.ternarize_parameter()
        # gamma comes from _compute_gamma with keepdim=True → (out, 1);
        # our eval buffer is (out,) so squeeze before copy_.
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

        return TernairLinearStorage(
            packed=self.packed_weight.cpu().numpy(),
            shape=self._packed_shape,
            gamma=self.gamma_eval.cpu().numpy().copy(),
            mode=self._pack_kind,
        )

    def is_frozen(self) -> bool:
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
        """Dequantise les poids empaquetes pour l'inference en mode eval().

        Corrige pour garantir que tous les tenseurs restent sur le
        meme device (ex: cuda:0) -- evite les erreurs de device mismatch
        lorsque le modele a ete deplace sur GPU.
        """
        if self._pack_kind is None or self._packed_shape is None:
            raise RuntimeError(
                "Call freeze_storage() before using the ternarised forward."
            )

        # 1. Device dynamique : suivre le device du module
        device = next(self.parameters()).device

        # 2. Decompacter les trits sur le bon device
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

        # 3. Gamma sur le meme device
        gamma = self.gamma_eval.to(dtype=dtype, device=device).unsqueeze(-1)

        # 4. Tout sur le device cible
        trits_tensor = trits_flat.to(dtype=dtype).reshape(self._packed_shape)
        return trits_tensor * gamma

    def forward(self, x: Tensor) -> Tensor:
        if self.training or self._pack_kind is None:
            return self._training_forward(x)
        return self._eval_forward(x)

    # ------------------------------------------------------------------
    # Storage accounting
    # ------------------------------------------------------------------
    def bytes_per_value(self) -> float | None:
        if self._pack_kind is None:
            return None
        if self._pack_kind == MODE_PACKED:
            return 8.0 / 5.0
        if self._pack_kind == MODE_FASTPACKED:
            return 2.0  # 2 bits/value
        return 8.0

    def state_bytes(self) -> int:
        """Bytes actually used for ternary storage (weights + γ only)."""
        if self._pack_kind is None:
            return 0
        packed_bytes = int(self.packed_weight.numel())
        # γ is FP32, one per output row.
        gamma_bytes = self.gamma_eval.numel() * 4
        return packed_bytes + gamma_bytes


__all__ = ["TernairLinear", "TernairLinearStorage"]
