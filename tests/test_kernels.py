"""Tests for the fastpacked ternary kernels."""

from __future__ import annotations

import numpy as np
import pytest

from ternair.kernels.packing_fast import (
    pack_trits_2bit,
    unpack_trits_2bit,
    trit_from_2bit,
    bits_from_trit,
)
from ternair.kernels.packed_ops import (
    decode_fastpacked_row,
    ternary_matmul_numpy,
    ternary_matmul_numpy_batched,
)


# ---------------------------------------------------------------------------
# 2-bit packing round-trip
# ---------------------------------------------------------------------------

def test_trit_from_2bit_correct() -> None:
    assert trit_from_2bit(0) == 0
    assert trit_from_2bit(1) == 1
    assert trit_from_2bit(2) == -1
    assert trit_from_2bit(3) == 0


def test_bits_from_trit_correct() -> None:
    assert bits_from_trit(0) == 0
    assert bits_from_trit(1) == 1
    assert bits_from_trit(-1) == 2


def test_pack_unpack_round_trip_random() -> None:
    rng = np.random.default_rng(0)
    for n in [1, 3, 4, 7, 32, 63, 128, 1007]:
        trits = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n)
        packed = pack_trits_2bit(trits)
        back = unpack_trits_2bit(packed, length=n)
        assert back.dtype == np.int8
        assert np.array_equal(back, trits), f"mismatch at n={n}"


def test_pack_unpack_round_trip_extremes() -> None:
    for trits in [
        np.array([-1, -1, -1, -1], dtype=np.int8),
        np.array([1, 1, 1, 1], dtype=np.int8),
        np.array([0, 0, 0, 0], dtype=np.int8),
        np.array([1, -1, 0, 1], dtype=np.int8),
    ]:
        packed = pack_trits_2bit(trits)
        back = unpack_trits_2bit(packed, length=len(trits))
        assert np.array_equal(back, trits)


def test_pack_ceil_bytes() -> None:
    assert pack_trits_2bit(np.zeros(4, dtype=np.int8)).shape == (1,)
    assert pack_trits_2bit(np.zeros(5, dtype=np.int8)).shape == (2,)
    assert pack_trits_2bit(np.zeros(8, dtype=np.int8)).shape == (2,)
    assert pack_trits_2bit(np.zeros(9, dtype=np.int8)).shape == (3,)


# ---------------------------------------------------------------------------
# decode_fastpacked_row matches unpack_trits_2bit
# ---------------------------------------------------------------------------

def test_decode_row_matches_unpack() -> None:
    rng = np.random.default_rng(1)
    trits = rng.choice([-1, 0, 1], size=100).astype(np.int8)
    packed = pack_trits_2bit(trits)
    decoded = decode_fastpacked_row(packed, N=100)
    assert np.array_equal(decoded, trits)


# ---------------------------------------------------------------------------
# ternary_matmul_numpy matches brute-force dot product
# ---------------------------------------------------------------------------

def test_matmul_numpy_exact() -> None:
    rng = np.random.default_rng(2)
    M, N = 4, 16
    trits = rng.choice([-1, 0, 1], size=(M, N)).astype(np.int8)
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])
    x = rng.random(N).astype(np.float16)
    gamma = rng.random(M).astype(np.float32)

    # Expected: brute-force unpack then dot
    expected = np.zeros(M, dtype=np.float32)
    for m in range(M):
        expected[m] = gamma[m] * np.dot(trits[m].astype(np.float32), x.astype(np.float32))

    result = ternary_matmul_numpy(packed, x, gamma)
    assert np.allclose(result.astype(np.float32), expected, atol=1e-3)


def test_matmul_numpy_batched_matches_single() -> None:
    rng = np.random.default_rng(3)
    M, N, B = 4, 12, 3
    trits = rng.choice([-1, 0, 1], size=(M, N)).astype(np.int8)
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])
    x_batch = rng.random((B, N)).astype(np.float16)
    gamma = rng.random(M).astype(np.float32)

    batched = ternary_matmul_numpy_batched(packed, x_batch, gamma)
    for b in range(B):
        single = ternary_matmul_numpy(packed, x_batch[b], gamma)
        assert np.allclose(batched[b].astype(np.float32), single.astype(np.float32), atol=1e-3)


# ---------------------------------------------------------------------------
# Triton matmul fallback behaviour
# ---------------------------------------------------------------------------

def test_triton_matmul_fallback() -> None:
    """Even without Triton, the fallback should give an identical result."""
    from ternair.kernels.triton_matmul import ternary_matmul_triton

    rng = np.random.default_rng(4)
    M, N = 4, 12
    trits = rng.choice([-1, 0, 1], size=M * N).astype(np.int8).reshape(M, N)
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])
    x = rng.random(N).astype(np.float16)
    gamma = rng.random(M).astype(np.float32)

    result = ternary_matmul_triton(packed, x, gamma)
    expected = ternary_matmul_numpy(packed, x, gamma)
    assert np.allclose(result.astype(np.float32), expected.astype(np.float32), atol=1e-3)


# ---------------------------------------------------------------------------
# C++ matmul fallback behaviour
# ---------------------------------------------------------------------------

def test_cpp_matmul_fallback() -> None:
    """Without cppyy, the fallback should match numpy."""
    from ternair.kernels.cpu_matmul import ternary_matmul_cpp

    rng = np.random.default_rng(5)
    M, N = 4, 12
    trits = rng.choice([-1, 0, 1], size=M * N).astype(np.int8).reshape(M, N)
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])
    x = rng.random(N).astype(np.float16)
    gamma = rng.random(M).astype(np.float32)

    result = ternary_matmul_cpp(packed, x, gamma)
    expected = ternary_matmul_numpy(packed, x, gamma)
    assert np.allclose(result.astype(np.float32), expected.astype(np.float32), atol=1e-3)


# ---------------------------------------------------------------------------
# fastpacked mode integration with TernairLinear
# ---------------------------------------------------------------------------

def test_ternair_linear_fastpacked_freeze_and_forward() -> None:
    import torch

    from ternair.quantization.linear import TernairLinear

    torch.manual_seed(6)
    layer = TernairLinear(8, 4, bias=False, storage="fastpacked")
    x = torch.randn(1, 3, 8)

    # Training forward — should work as usual
    out_train = layer(x)
    assert out_train.shape == (1, 3, 4)

    # Freeze — switch to fastpacked storage
    s = layer.freeze_storage()
    assert s.mode == "fastpacked"

    # Eval forward — should work
    layer.eval()
    out_eval = layer(x)
    assert out_eval.shape == (1, 3, 4)

    # Both results should be close (not identical since training uses
    # a slightly different γ pathway).
    mse = float(((out_train - out_eval) ** 2).mean())
    assert mse < 1.0, f"MSE too large: {mse:.4f}"
