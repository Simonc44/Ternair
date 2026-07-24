"""Tests for the ThalamicBottleneck compression module."""

from __future__ import annotations

import pytest

_HAS_TORCH = True
try:
    import torch  # noqa: F401
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch required")


@pytest.fixture
def config():
    from ternair import TernairConfig
    return TernairConfig(
        hidden_size=128,
        storage="packed",
        num_hidden_layers=4,
        thalamus_k=32,
        thalamus_heads=4,
    )


def test_thalamus_compresses_variable_input(config) -> None:
    from ternair.model.thalamus import ThalamicBottleneck

    tb = ThalamicBottleneck(config, input_dim=128)

    # Input shapes: (B, N, D) with different N
    for N in [16, 64, 196, 512, 1024]:
        x = torch.randn(2, N, 128)
        out = tb(x)
        assert out.shape == (2, 32, 128), f"Failed at N={N}: got {out.shape}"
        assert torch.isfinite(out).all()


def test_thalamus_projection_dim_mismatch(config) -> None:
    from ternair.model.thalamus import ThalamicBottleneck

    # Input dim differs from hidden
    tb = ThalamicBottleneck(config, input_dim=256)
    x = torch.randn(2, 64, 256)
    out = tb(x)
    assert out.shape == (2, 32, 128)


def test_thalamus_forward_backward(config) -> None:
    from ternair.model.thalamus import ThalamicBottleneck

    tb = ThalamicBottleneck(config, input_dim=128)
    x = torch.randn(2, 64, 128)

    # Forward
    out = tb(x)
    loss = out.sum()

    # Backward
    loss.backward()

    # All TernairLinear parameters should have gradients
    for name, param in tb.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_thalamus_with_tiny_profile() -> None:
    from ternair import TernairForCausalLM, tiny_profile

    # tiny profile + thalamus config
    cfg = tiny_profile(storage="packed")
    cfg.thalamus_k = 32
    cfg.thalamus_heads = 4
    model = TernairForCausalLM(cfg)

    # The model forward should still work (thalamus is optional, not used
    # unless explicit in the forward path)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(ids)
    assert out.shape == (1, 8, cfg.vocab_size)
