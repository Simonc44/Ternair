"""T-LoRA / BitDelta — Ternary Low-Rank Adaptation fine-tuning.

Allows adding lightweight ternary adapters to a frozen Ternair model
for task-specific specialisation without modifying the base weights.

BitDelta
--------
Delta weights DW are ternarised: DW in {-1, 0, +1}.
Each adapter stores only the ternary delta plus a tiny per-channel scale.
Total cost: ~1.6 bits per delta parameter.

T-LoRA
------
DW is decomposed into low-rank matrices A x B where A, B are ternary.
For hidden_size H and rank r: H*r + r*H ternary parameters vs H*H.
Typical r=8 or r=16 achieves >90% parameter reduction.

Usage:
    adapter = TernaryLoRAAdapter(hidden_size=256, rank=8)
    model.register_adapter(adapter, layer_ids=[0, 5, 10])
    # Train only adapter params, base model frozen
    output = model(input_ids)  # automatically applies adapters
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.quantization.ternary import _compute_gamma, ternarize_ste
from ternair.quantization.linear import TernairLinear


# ---------------------------------------------------------------------------
# Ternary LoRA Adapter
# ---------------------------------------------------------------------------

class TernaryLoRALinear(nn.Module):
    """Low-rank ternary adapter applied to one linear layer.

    Forward: x -> base(x) + scale * ternary(A @ B) @ x

    A and B are low-rank matrices stored in FP but ternarised during
    forward via STE, so they learn ternary-compatible values.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)

        # Low-rank factor A (ternary): (out_features, rank)
        self.lora_A = nn.Parameter(torch.empty(out_features, rank))
        # Low-rank factor B (ternary): (rank, in_features)
        self.lora_B = nn.Parameter(torch.empty(rank, in_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Per-channel scales for the ternary factors
        self.register_buffer(
            "gamma_A", torch.ones(out_features, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "gamma_B", torch.ones(rank, dtype=torch.float32), persistent=True
        )

    def forward(self, x: Tensor) -> Tensor:
        # Ternarise both factors with STE
        w_a_eff, _ = ternarize_ste(self.lora_A, dim=-1)
        w_b_eff, _ = ternarize_ste(self.lora_B, dim=-1)

        # Compute delta = ternary(A) @ ternary(B) (both ternarised)
        # delta shape: (out_features, in_features)
        delta = torch.matmul(
            w_a_eff.to(x.dtype),
            w_b_eff.to(x.dtype),
        )

        # Apply adapter: x -> x + scale * (delta @ x^T)^T
        x = self.dropout(x)
        return self.scale * F.linear(x, delta)


# ---------------------------------------------------------------------------
# BitDelta: Simple ternary delta weights
# ---------------------------------------------------------------------------

class TernaryDeltaLinear(nn.Module):
    """Full-rank ternary delta adapter.

    DW is a ternary delta of the same shape as the base weight.
    DW in {-1, 0, +1} with one scale per output row.
    ~1.6 bits per parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.scale = alpha / max(in_features, 1)

        self.delta = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer(
            "gamma_delta", torch.ones(out_features, dtype=torch.float32),
            persistent=True,
        )
        nn.init.zeros_(self.delta)

    def forward(self, x: Tensor) -> Tensor:
        w_d_eff, _ = ternarize_ste(self.delta, dim=-1)
        return self.scale * F.linear(x, w_d_eff.to(x.dtype))


# ---------------------------------------------------------------------------
# Adapter registry — attach adapters to a model
# ---------------------------------------------------------------------------

class AdapterRegistry:
    """Manages ternary adapters attached to a base model.

    Example:
        registry = AdapterRegistry()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                adapter = TernaryLoRALinear(
                    module.in_features, module.out_features, rank=8
                )
                registry.register(name, adapter)
    """

    def __init__(self) -> None:
        self._adapters: dict[str, nn.Module] = {}

    def register(self, target_name: str, adapter: nn.Module) -> None:
        """Register an adapter for a target module name."""
        self._adapters[target_name] = adapter

    def forward_hook(self, name: str):
        """Create a forward hook that applies the adapter."""
        def hook(module, input, output):
            adapter = self._adapters.get(name)
            if adapter is not None:
                return output + adapter(input[0])
            return output
        return hook

    def attach(self, model: nn.Module) -> None:
        """Attach all registered adapters to the model via forward hooks."""
        self._handles = []
        for name, module in model.named_modules():
            if name in self._adapters:
                handle = module.register_forward_hook(self.forward_hook(name))
                self._handles.append(handle)

    def detach(self) -> None:
        """Remove all adapter hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def adapter_params(self) -> list[nn.Parameter]:
        """Return all trainable adapter parameters."""
        params = []
        for adapter in self._adapters.values():
            params.extend(p for p in adapter.parameters() if p.requires_grad)
        return params

    def count_params(self) -> int:
        """Count total adapter parameters."""
        return sum(p.numel() for p in self.adapter_params())

    def state_dict(self) -> dict:
        """Return adapter state dict for saving."""
        state = {}
        for name, adapter in self._adapters.items():
            state[name] = adapter.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load adapter state dict."""
        for name, adapter_state in state.items():
            if name in self._adapters:
                self._adapters[name].load_state_dict(adapter_state)


# ---------------------------------------------------------------------------
# Convenience: create adapters for all linear layers in a model
# ---------------------------------------------------------------------------

def add_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
    exclude_modules: Optional[list[str]] = None,
) -> AdapterRegistry:
    """Add ternary LoRA adapters to all (or specified) linear layers.

    Args:
        model: The base model.
        rank: LoRA rank.
        alpha: LoRA scaling alpha.
        target_modules: List of module name substrings to target.
            If None, targets all nn.Linear and TernairLinear.
        exclude_modules: Module name substrings to skip.

    Returns:
        AdapterRegistry with the created adapters.
    """
    if target_modules is None:
        target_modules = [""]  # matches everything

    if exclude_modules is None:
        exclude_modules = ["embed", "lm_head", "norm", "ln_"]

    from ternair.quantization.linear import TernairLinear

    registry = AdapterRegistry()

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, TernairLinear)):
            continue
        if any(excl in name.lower() for excl in exclude_modules):
            continue
        if not any(tgt in name.lower() for tgt in target_modules):
            continue

        adapter = TernaryLoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
        )
        registry.register(name, adapter)

    registry.attach(model)
    print(
        f"Added {len(registry._adapters)} LoRA adapters "
        f"(rank={rank}, {registry.count_params():,} params)"
    )
    return registry


def add_bitdelta_to_model(
    model: nn.Module,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
    exclude_modules: Optional[list[str]] = None,
) -> AdapterRegistry:
    """Add BitDelta (full-rank ternary delta) adapters.

    Args:
        model: The base model.
        alpha: Delta scaling alpha.
        target_modules: Module name substrings to target.
        exclude_modules: Module name substrings to skip.

    Returns:
        AdapterRegistry with the created adapters.
    """
    if target_modules is None:
        target_modules = [""]
    if exclude_modules is None:
        exclude_modules = ["embed", "lm_head", "norm", "ln_"]

    from ternair.quantization.linear import TernairLinear

    registry = AdapterRegistry()

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, TernairLinear)):
            continue
        if any(excl in name.lower() for excl in exclude_modules):
            continue
        if not any(tgt in name.lower() for tgt in target_modules):
            continue

        adapter = TernaryDeltaLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            alpha=alpha,
        )
        registry.register(name, adapter)

    registry.attach(model)
    print(
        f"Added {len(registry._adapters)} BitDelta adapters "
        f"({registry.count_params():,} params)"
    )
    return registry


__all__ = [
    "TernaryLoRALinear",
    "TernaryDeltaLinear",
    "AdapterRegistry",
    "add_lora_to_model",
    "add_bitdelta_to_model",
]
