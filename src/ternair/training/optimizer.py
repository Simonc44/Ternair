"""Optimiser factory with decoupled weight decay for ternary models.

Three parameter groups are created:

1. **TernairLinear.weight** — zero weight decay, *full* LR applied
   (the STE gradient already constrains; WD on the master FP weights
   would push them toward zero, harming the ternary representation).
2. **Embeddings + LM head** — standard weight decay.
3. **All others** (RMSNorm γ, biases, gamma_eval buffers) — zero
   weight decay (standard convention for normalisation parameters).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ternair.quantization.linear import TernairLinear


def create_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    ternair_lr_scale: float = 1.0,
) -> list[dict]:
    """Build parameter groups with decoupled weight decay.

    Parameters
    ----------
    model
        A :class:`TernairForCausalLM` (or any ``nn.Module`` with
        :class:`TernairLinear` children).
    lr
        Base learning rate.
    weight_decay
        Weight decay applied to non-ternary, non-norm, non-bias params.
    ternair_lr_scale
        Scale factor for the ternary linear parameters' LR (default 1.0
        — full LR).  Some recipes lower this to 0.5× during early
        training.

    Returns
    -------
    param_groups
        List of dicts suitable for ``torch.optim.AdamW``.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    ternary_params: list[nn.Parameter] = []

    for name, module in model.named_modules():
        is_ternary_linear = isinstance(module, TernairLinear)

        for pname, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            # Bias → no decay
            if pname == "bias" and param is not None:
                no_decay_params.append(param)
                continue

            # TernairLinear weight → separate group
            if is_ternary_linear and pname == "weight":
                ternary_params.append(param)
                continue

            # RMSNorm / LayerNorm weight → no decay
            if isinstance(module, (nn.LayerNorm,)) or "norm" in name.lower() and pname == "weight":
                no_decay_params.append(param)
                continue

            # Embedding → decay
            if isinstance(module, nn.Embedding):
                decay_params.append(param)
                continue

            # Everything else → decay by default, but skip if it's a norm-like
            if ".weight" in name and any(nm in name.lower() for nm in ["norm", "ln_", "layernorm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        {
            "params": ternary_params,
            "lr": lr * ternair_lr_scale,
            "weight_decay": 0.0,
        },
    ]

    # Filter out empty groups
    return [g for g in groups if g["params"]]


def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    ternair_lr_scale: float = 1.0,
) -> torch.optim.AdamW:
    """Convenience: create an AdamW optimiser with decoupled WD groups."""
    groups = create_param_groups(
        model, lr=lr, weight_decay=weight_decay, ternair_lr_scale=ternair_lr_scale
    )
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def clip_gradients(
    model: nn.Module, max_norm: float = 1.0
) -> float:
    """Clip gradients, returning the total norm before clipping."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    return float(total_norm.item())
