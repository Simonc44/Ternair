"""Smoke test for the full training step."""

from __future__ import annotations

import pytest


@pytest.fixture
def tiny_model():
    import torch
    from ternair import TernairForCausalLM, tiny_profile

    torch.manual_seed(42)
    return TernairForCausalLM(tiny_profile(storage="packed"))


def test_cross_entropy_returns_finite(tiny_model):
    import torch
    from ternair.training.trainer import cross_entropy

    ids = torch.randint(0, 4096, (2, 16))
    logits = tiny_model(ids)
    loss = cross_entropy(logits, ids)
    assert torch.isfinite(loss)


def test_optimizer_step_reduces_loss(tiny_model):
    import torch
    from ternair.training.optimizer import create_optimizer
    from ternair.training.trainer import cross_entropy

    ids = torch.randint(0, 4096, (2, 16))
    optimizer = create_optimizer(tiny_model, lr=1e-3, weight_decay=0.0)

    tiny_model.train()
    logits = tiny_model(ids)
    loss1 = cross_entropy(logits, ids)

    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()

    logits = tiny_model(ids)
    loss2 = cross_entropy(logits, ids)

    # Loss should go down after one step on the same data (overfit)
    assert loss2.item() < loss1.item()


def test_gradient_flow_through_ste(tiny_model):
    import torch
    from ternair.training.trainer import cross_entropy
    from ternair.quantization.linear import TernairLinear

    ids = torch.randint(0, 4096, (2, 16))
    logits = tiny_model(ids)
    loss = cross_entropy(logits, ids)
    loss.backward()

    # Every TernairLinear weight should have a finite gradient
    for module in tiny_model.modules():
        if isinstance(module, TernairLinear):
            assert module.weight.grad is not None, "No gradient for ternary weight"
            assert torch.isfinite(module.weight.grad).all(), "Non-finite gradient"
