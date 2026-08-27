"""Shared RMSNorm implementation.

Extracted from :mod:`ternair.model.block` so that attention and MLP
modules (which are imported *by* the block) can also use RMSNorm without
creating a circular import.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Compute in fp32 for numerical stability but return in the input
        # dtype (fp16 / bf16 for inference) -- ``x`` is reassigned below,
        # so capture the input dtype *before* the fp32 upcast.
        dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(var + self.eps)
        return (x * self.weight).to(dtype)


__all__ = ["RMSNorm"]
