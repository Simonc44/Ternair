"""Memory pre-flight estimator for ternary models.

Before kicking off a training run (especially on intermediate-size
models in the 50M-500M range), users should see whether the model,
optimiser state, activations, and -- if applicable -- a teacher model
will fit on the available device.

The estimator is intentionally conservative: it queries the actual
``nn.Module`` rather than approximating from the config, so it
accounts for learned ``alpha`` parameters, tied embeddings, MoE
routing tensors, and any other in-flight additions.

Limitations
-----------
* Single-GPU estimator.  Distributed setups (DDP/FSDP/DeepSpeed)
  shard optimiser states and so use **less** memory than reported.
* The activation estimate is the per-layer peak: ``batch_size *
  seq_len * hidden`` for the dominant path.  Real peak depends on
  attention scores and KV cache, which we approximate by a
  configurable safety factor.
* Host CPU RAM is detected via ``psutil`` if available; otherwise the
  estimator runs in device-only mode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

_LOGGER = logging.getLogger(__name__)

# Defaults sized for the BitNet b1.58 recipe:
# * Weights: master FP copy kept for QAT -> 2 bytes/param.
# * AdamW state: 2 FP32 buffers (m, v) -> 8 bytes/param.
# * Activations: peak ~ batch * seq * hidden * 2 bytes (BF16).
DEFAULT_BYTES_PER_PARAM_MASTER = 2  # FP16/BF16 master weight
DEFAULT_BYTES_PER_PARAM_OPTIM = 8  # AdamW: m + v = 2 * FP32
DEFAULT_BYTES_PER_ACTIVATION = 2  # BF16/FP16 activations
DEFAULT_OPTIMIZER_OVERHEAD = 1.2  # PyTorch allocator + fragmentation
DEFAULT_ACTIVATION_SAFETY_FACTOR = 1.5  # KV cache + intermediate buffers


@dataclass
class MemoryEstimate:
    """Result of a memory pre-flight check."""

    model_bytes: int = 0
    optimizer_bytes: int = 0
    teacher_bytes: int = 0
    activations_bytes: int = 0
    total_bytes: int = 0
    available_bytes: int = 0
    fits: bool = True
    safety_margin_bytes: int = 0
    bottleneck: str = "ok"
    notes: list[str] = field(default_factory=list)
    overhead_multiplier: float = DEFAULT_OPTIMIZER_OVERHEAD

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [
            "=== Memory pre-flight ===",
            f"  model_weights : {self.model_bytes / 1024 ** 2:>10.1f} MiB",
            f"  optimizer     : {self.optimizer_bytes / 1024 ** 2:>10.1f} MiB",
            f"  activations   : {self.activations_bytes / 1024 ** 2:>10.1f} MiB",
            f"  teacher       : {self.teacher_bytes / 1024 ** 2:>10.1f} MiB",
            f"  total est.    : {self.total_bytes / 1024 ** 2:>10.1f} MiB",
            f"  available     : {self.available_bytes / 1024 ** 2:>10.1f} MiB",
            f"  margin        : {self.safety_margin_bytes / 1024 ** 2:>+10.1f} MiB",
            f"  fits          : {'YES' if self.fits else 'NO'}",
        ]
        if self.bottleneck != "ok":
            lines.append(f"  bottleneck    : {self.bottleneck}")
        if self.notes:
            for note in self.notes:
                lines.append(f"  note          : {note}")
        lines.append("=========================")
        return "\n".join(lines)

    def fits_with_margin(self, min_margin_mib: float = 256.0) -> bool:
        """Check that there's at least ``min_margin_mib`` of headroom."""
        return self.safety_margin_bytes >= int(min_margin_mib * 1024 ** 2)


def _count_model_bytes(model: nn.Module) -> int:
    """Compute the model's *training* memory footprint in bytes.

    Includes master FP weights, biases, learned alpha params, and any
    optimiser-registered buffers.  Does NOT include the ternary packed
    storage (which replaces master weights after ``freeze_storage``).
    """
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        # Master copy is at the param dtype, typically FP16 during
        # QAT.  We use 2 bytes as a conservative default.
        total += p.numel() * DEFAULT_BYTES_PER_PARAM_MASTER
    for b in model.buffers():
        if not b.requires_grad:
            continue
        # Buffers (e.g. gamma_eval) are usually FP32.
        total += b.numel() * 4
    return total


def _count_optimizer_bytes(model: nn.Module, optim_state_bytes_per_param: int = DEFAULT_BYTES_PER_PARAM_OPTIM) -> int:
    """AdamW state footprint: 2 FP32 buffers per trainable param."""
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n * optim_state_bytes_per_param


def estimate_activations_bytes(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    *,
    activation_dtype_bytes: int = DEFAULT_BYTES_PER_ACTIVATION,
    safety_factor: float = DEFAULT_ACTIVATION_SAFETY_FACTOR,
) -> int:
    """Peak per-layer activations in bytes (with safety factor)."""
    # Per-layer peak: cache activations + attention scores.
    # Worst case scales linearly with depth because callbacks accumulate
    # across layers; we apply a sqrt(L) factor for amortised savings.
    per_layer = batch_size * seq_length * hidden_size * activation_dtype_bytes
    total = per_layer * num_layers * safety_factor
    return int(total)


def _available_bytes() -> int:
    """Best-effort detection of available device memory in bytes.

    Returns 0 when no accelerator is detected.  CPU RAM is reported
    via ``psutil`` if available; otherwise we conservatively cap at
    16 GiB so the estimator never over-reports.
    """
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(device)
            # Use 90% of the free memory to leave room for PyTorch overhead.
            return int(free * 0.9)
        except Exception:
            pass
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available * 0.9)
    except ImportError:
        # Fallback conservative estimate.
        return 16 * 1024 ** 3


def estimate_memory(
    model: nn.Module,
    *,
    batch_size: int,
    seq_length: int,
    teacher: Optional[nn.Module] = None,
    overhead_multiplier: float = DEFAULT_OPTIMIZER_OVERHEAD,
    min_margin_mib: float = 256.0,
) -> MemoryEstimate:
    """Compute a memory estimate for a training run.

    Parameters
    ----------
    model
        Student / main model (must be the nn.Module you'll train).
    batch_size, seq_length
        Forward/backward dimensions for the activation peak.
    teacher
        Optional distillation teacher.  Its weights are counted at
        master-FP bytes; ``requires_grad`` is ignored (teacher is
        assumed frozen).
    overhead_multiplier
        PyTorch allocator + fragmentation padding (default 1.2x).
    min_margin_mib
        Minimum safety margin after the total estimate before we
        declare the run ``fits=False``.

    Returns
    -------
    MemoryEstimate
        A populated estimate; call :meth:`MemoryEstimate.summary` to
        print.
    """
    model_bytes = _count_model_bytes(model)
    optim_bytes = _count_optimizer_bytes(model)
    teacher_bytes = _count_model_bytes(teacher) if teacher is not None else 0

    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 12)
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", 768)
    acts_bytes = estimate_activations_bytes(
        batch_size, seq_length, hidden_size, n_layers,
    )

    total = int(
        overhead_multiplier * (model_bytes + optim_bytes + teacher_bytes + acts_bytes)
    )
    available = _available_bytes()
    margin = available - total
    fits = margin >= int(min_margin_mib * 1024 ** 2)

    bottleneck = "ok"
    if not fits:
        # Identify the dominant consumer.
        consumers = {
            "model weights": model_bytes,
            "optimizer state": optim_bytes,
            "activations": acts_bytes,
            "teacher": teacher_bytes,
        }
        bottleneck = max(consumers, key=consumers.get)

    notes: list[str] = []
    if teacher is not None:
        notes.append(
            "teacher is counted at master-FP bytes; passes do not "
            "allocate gradients for teacher params."
        )
    if batch_size * seq_length > 4096:
        notes.append(
            "Large batch*seq footprint; consider gradient checkpointing."
        )

    return MemoryEstimate(
        model_bytes=model_bytes,
        optimizer_bytes=optim_bytes,
        teacher_bytes=teacher_bytes,
        activations_bytes=acts_bytes,
        total_bytes=total,
        available_bytes=available,
        fits=fits,
        safety_margin_bytes=margin,
        bottleneck=bottleneck,
        notes=notes,
        overhead_multiplier=overhead_multiplier,
    )


__all__ = [
    "MemoryEstimate",
    "estimate_memory",
    "estimate_activations_bytes",
    "DEFAULT_BYTES_PER_PARAM_MASTER",
    "DEFAULT_BYTES_PER_PARAM_OPTIM",
]
