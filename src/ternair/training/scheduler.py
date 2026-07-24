"""WSD (Warmup-Stable-Decay) learning rate scheduler.

The schedule is defined by three ratios of the total training steps:

1. **Warmup** (``warmup_ratio``) — linear increase from 0 to peak LR.
2. **Stable**  (``stable_ratio``) — constant peak LR.
3. **Decay**   (``decay_ratio``) — cosine or linear decay to ``min_lr``.

The ratios are normalised to sum to ``≤ 1.0``; any leftover steps
(i.e. ``1 - warmup - stable - decay``) are treated as stable.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch.optim.lr_scheduler import LRScheduler

DecayType = Literal["cosine", "linear"]


class WSDScheduler(LRScheduler):
    """Warmup-Stable-Decay scheduler.

    Parameters
    ----------
    optimizer
        Wrapped PyTorch optimizer.
    total_steps
        Total number of training steps.
    warmup_steps
        Number of steps for the linear warmup phase.
    stable_steps
        Number of steps at the constant peak LR.
    decay_steps
        Number of steps for the decay phase.  If ``warmup + stable + decay
        < total_steps``, the remaining steps stay at the peak LR.
    min_lr
        LR at the very end of the decay.
    decay_type
        ``"cosine"`` or ``"linear"``.
    last_epoch
        Passed to :class:`torch.optim.lr_scheduler.LRScheduler`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 0.0,
        decay_type: DecayType = "cosine",
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.decay_type = decay_type
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch  # 0-indexed
        base_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]  # type: ignore[union-attr]

        if step < self.warmup_steps:
            # Linear warmup
            factor = float(step) / max(self.warmup_steps, 1)
            return [lr * factor for lr in base_lrs]

        if step < self.warmup_steps + self.stable_steps:
            # Stable — constant peak LR
            return list(base_lrs)

        if step < self.total_steps and self.decay_steps > 0:
            # Decay phase
            decay_progress = float(step - self.warmup_steps - self.stable_steps) / max(self.decay_steps, 1)
            decay_progress = min(decay_progress, 1.0)
            if self.decay_type == "cosine":
                factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            else:  # linear
                factor = 1.0 - decay_progress
            return [self.min_lr + (lr - self.min_lr) * factor for lr in base_lrs]

        # Past the end — stay at min_lr
        return [self.min_lr] * len(base_lrs)
