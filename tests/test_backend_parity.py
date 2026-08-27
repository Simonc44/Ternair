"""Backend parity tests: verify PyTorch and NumPy backends agree."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ternair.kernels.packing_fast import pack_trits_2bit
from ternair.kernels.packed_ops import ternary_matmul_numpy


def test_parity_pytorch_vs_numpy_matmul():
    """Ternary matmul: PyTorch dequantise vs NumPy packed path."""
    rng = np.random.default_rng(42)
    M, N = 8, 32
    trits = rng.choice([-1, 0, 1], size=(M, N)).astype(np.int8)

    # Pack
    packed = np.zeros((M, (N + 3) // 4), dtype=np.uint8)
    for m in range(M):
        packed[m] = pack_trits_2bit(trits[m])

    x = rng.random(N).astype(np.float16)
    gamma = rng.random(M).astype(np.float32)

    # NumPy reference
    np_result = ternary_matmul_numpy(packed, x, gamma)

    # PyTorch dequantise
    from ternair.kernels.packing_fast import unpack_trits_2bit

    flat = unpack_trits_2bit(packed.ravel(), length=M * N)
    w = torch.from_numpy(flat).reshape(M, N).float()
    x_t = torch.from_numpy(x.astype(np.float32))
    g_t = torch.from_numpy(gamma)
    pt_result = (w @ x_t * g_t).numpy()

    assert np.allclose(np_result.astype(np.float32), pt_result, atol=1e-3), (
        f"Max diff: {np.abs(np_result.astype(np.float32) - pt_result).max():.6f}"
    )


def test_parity_frozen_model_deterministic():
    """Same frozen model should produce identical logits across runs."""
    from ternair import TernairForCausalLM, tiny_profile

    cfg = tiny_profile(storage="packed")
    model = TernairForCausalLM(cfg)
    model.freeze_storage()
    model.eval()

    ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        out1 = model(ids)
        out2 = model(ids)

    assert torch.allclose(out1, out2, atol=0.0), "Frozen model must be bit-exact"


def test_parity_different_seeds_different_outputs():
    """Different random seeds should produce different model outputs."""
    from ternair import TernairForCausalLM, tiny_profile

    cfg = tiny_profile(storage="packed")

    torch.manual_seed(0)
    m1 = TernairForCausalLM(cfg)
    m1.freeze_storage()
    m1.eval()

    torch.manual_seed(1)
    m2 = TernairForCausalLM(cfg)
    m2.freeze_storage()
    m2.eval()

    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out1 = m1(ids)
        out2 = m2(ids)

    assert not torch.equal(out1, out2), "Different seeds should yield different outputs"
