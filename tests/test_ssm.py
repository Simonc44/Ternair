"""Tests for the TernarySSM block and hybrid model."""

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
        hidden_size=64,
        storage="packed",
        num_hidden_layers=4,
        num_attn_layers=2,  # 2 attn + 2 SSM layers
        ssm_dim=8,
    )


def test_ssm_forward_shape(config) -> None:
    from ternair.model.ssm import TernarySSMBlock

    block = TernarySSMBlock(config)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


def test_ssm_runs_on_different_seq_lengths(config) -> None:
    from ternair.model.ssm import TernarySSMBlock

    block = TernarySSMBlock(config)
    for L in [1, 3, 8, 32, 128]:
        x = torch.randn(1, L, 64)
        out = block(x)
        assert out.shape == (1, L, 64), f"Failed at L={L}"


def test_hybrid_model_forward(config) -> None:
    from ternair import TernairForCausalLM

    model = TernairForCausalLM(config)
    assert len(model.model.layers) == 4

    ids = torch.randint(0, config.vocab_size, (1, 16))
    out = model(ids)
    assert out.shape == (1, 16, config.vocab_size)
    assert torch.isfinite(out).all()


def test_hybrid_model_generate(config) -> None:
    from ternair import TernairForCausalLM, generate

    model = TernairForCausalLM(config)
    # freeze + generate
    snapshot = model.freeze_storage()
    model.eval()

    ids = torch.randint(0, config.vocab_size, (1, 8))
    out = generate(model, ids, max_new_tokens=4)
    assert out.shape == (1, 12)
    assert torch.isfinite(out).all()


def test_hybrid_model_mixed_layers(config) -> None:
    """Verify that the first config.num_attn_layers layers are attention,
    and the rest are SSM."""
    from ternair.model.hybrid_block import TernairHybridBlock

    for i in range(4):
        block = TernairHybridBlock(config, layer_idx=i)
        if i < 2:
            assert block.is_attn, f"Layer {i} should be attention"
        else:
            assert not block.is_attn, f"Layer {i} should be SSM"


def test_ssm_selective_scan_gradient_flow(config) -> None:
    from ternair.model.ssm import TernarySSMBlock

    block = TernarySSMBlock(config)
    x = torch.randn(1, 8, 64)

    out = block(x)
    loss = out.sum()
    loss.backward()

    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name}: no grad"
            assert torch.isfinite(param.grad).all(), f"{name}: non-finite grad"
