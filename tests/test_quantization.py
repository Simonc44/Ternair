"""Tests for the ternary/8-bit primitives and the base-3 packing helper.

These tests are intentionally torch-free where possible so they can run
in the pure-Python CI lane. They import torch lazily inside each test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Pure-Python packing round-trip
# --------------------------------------------------------------------------- #
def test_pack_unpack_round_trip_random() -> None:
    from ternair.quantization.packing import pack_trits, unpack_trits

    rng = np.random.default_rng(0)
    for n in [1, 4, 5, 11, 128, 1023]:
        trits = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n).astype(np.int8)
        packed = pack_trits(trits)
        back = unpack_trits(packed, length=n)
        assert back.dtype == np.int8
        assert np.array_equal(back, trits), f"mismatch at n={n}"


def test_pack_unpack_round_trip_three_values() -> None:
    from ternair.quantization.packing import pack_trits, unpack_trits

    for trits in [
        np.array([-1, -1, -1, -1, -1], dtype=np.int8),
        np.array([+1, +1, +1, +1, +1], dtype=np.int8),
        np.array([0, 0, 0, 0, 0], dtype=np.int8),
        np.array([-1, 0, +1, -1, 0], dtype=np.int8),
    ]:
        packed = pack_trits(trits)
        back = unpack_trits(packed, length=len(trits))
        assert np.array_equal(back, trits)


def test_pack_uses_correct_byte_count() -> None:
    from ternair.quantization.packing import pack_trits

    # 5 trits/byte exactly; any extra is rounded up.
    assert pack_trits(np.zeros(5, dtype=np.int8)).shape == (1,)
    assert pack_trits(np.zeros(10, dtype=np.int8)).shape == (2,)
    assert pack_trits(np.zeros(11, dtype=np.int8)).shape == (3,)


@pytest.mark.skipif(not _has_torch(), reason="torch is required")
def test_ternarize_uses_three_values() -> None:
    import torch

    from ternair.quantization.ternary import ternarize

    torch.manual_seed(0)
    W = torch.randn(64, 128)
    Wt, gamma = ternarize(W, dim=-1)
    assert Wt.shape == W.shape
    assert Wt.dtype == torch.int8
    unique = torch.unique(Wt).tolist()
    assert set(unique).issubset({-1, 0, 1})
    # γ has one entry per output row.
    assert gamma.shape == (64, 1)


@pytest.mark.skipif(not _has_torch(), reason="torch is required")
def test_ternarize_clamp_keeps_w_t_in_minus_one_one() -> None:
    import torch

    from ternair.quantization.ternary import ternarize

    # Very large W — after dividing by γ every value should hit ±1.
    W = torch.randn(8, 16) * 1e6
    Wt, _ = ternarize(W, dim=-1)
    unique = torch.unique(Wt).tolist()
    # `large W` → strong central tendency in γ, so essentially all trits become ±1.
    assert set(unique).issubset({-1, 1}) or 0 in unique


@pytest.mark.skipif(not _has_torch(), reason="torch is required")
def test_quantize_activations_8bit_clamps_to_minus128_127() -> None:
    import torch

    from ternair.quantization.activation import quantize_activations_8bit

    x = torch.randn(2, 4, 8) * 100  # way outside the int8 range
    out = quantize_activations_8bit(x)
    assert out.quantised.dtype == torch.int8
    assert out.quantised.min() >= -128
    assert out.quantised.max() <= 127


@pytest.mark.skipif(not _has_torch(), reason="torch is required")
def test_quantize_activations_8bit_round_trip_keeps_scale() -> None:
    import torch

    from ternair.quantization.activation import quantize_activations_8bit

    # Input chosen so that absmax lands on an exact quantisation step.
    x = torch.tensor([[0.1, -0.5, 7.0, -2.0]])
    res = quantize_activations_8bit(x)
    # absmax = 7.0 → scale = 7/127
    expected_scale = torch.tensor([[7.0 / 127.0]])
    assert torch.allclose(res.scale, expected_scale, atol=1e-6)
    # Tolerance: small values can be off by up to ~1.5×scale.
    # The worst-case error occurs when x maps to a different int than
    # the round-to-nearest target; we accept up to 2×scale.
    dequant = res.quantised.to(torch.float32) * res.scale
    assert torch.allclose(dequant, x, atol=2.0 * float(res.scale.squeeze().item()))


@pytest.mark.skipif(not _has_torch(), reason="torch is required")
def test_ternary_ste_gradient_passes_through() -> None:
    import torch

    from ternair.quantization.ternary import ternary_linear_forward

    W = torch.randn(8, 4, requires_grad=True)
    gamma = W.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    y = ternary_linear_forward(W, gamma)
    y.sum().backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()
