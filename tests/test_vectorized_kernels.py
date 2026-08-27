"""Regression tests for the vectorized packed kernels and cached dequantisation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ternair.kernels.packing_fast import pack_trits_2bit, unpack_trits_2bit
from ternair.kernels.packed_ops import (
    unpack_fastpacked_matrix,
    ternary_matmul_numpy,
    ternary_matmul_numpy_batched,
)


def test_unpack_fastpacked_matrix_matches_reference():
    """The vectorised unpack must match the per-byte reference decode."""
    rng = np.random.default_rng(0)
    M, N = 6, 20
    trits = rng.choice([-1, 0, 1], size=(M, N)).astype(np.int8)
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])

    W = unpack_fastpacked_matrix(packed)[:, :N]
    flat = unpack_trits_2bit(packed.ravel(), length=M * N).reshape(M, N)
    assert np.array_equal(W, flat)


def test_batched_numpy_matches_single():
    """Batched matmul must equal per-row single matmul."""
    rng = np.random.default_rng(1)
    M, N, B = 4, 16, 3
    trits = rng.choice([-1, 0, 1], size=(M, N)).astype(np.int8)
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])
    xb = rng.random((B, N)).astype(np.float16)
    gamma = rng.random(M).astype(np.float32)

    batched = ternary_matmul_numpy_batched(packed, xb, gamma)
    for b in range(B):
        single = ternary_matmul_numpy(packed, xb[b], gamma)
        assert np.allclose(batched[b], single, atol=1e-3)


def test_cached_dequantise_is_bit_exact_across_forwards():
    """After freeze, repeated forwards with the torch backend must be bit-exact."""
    from ternair import TernairForCausalLM, tiny_profile
    from ternair.quantization.linear import TernairLinear

    torch.manual_seed(0)
    model = TernairForCausalLM(tiny_profile(storage="packed"))
    model.freeze_storage()
    model.eval()
    for m in model.modules():
        if isinstance(m, TernairLinear):
            m.set_inference_backend("torch")

    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        o1 = model(ids)
        o2 = model(ids)
        o3 = model(ids)
    assert torch.equal(o1, o2)
    assert torch.equal(o2, o3)
