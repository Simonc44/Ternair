"""Tests for the vectorised parallel SSM scan.

Verifies that :func:`_selective_scan_parallel` produces numerically
identical results to a reference sequential scan for both CPU and
CUDA (CUDA test is skipped if unavailable).
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference sequential scan (ground truth)
# ---------------------------------------------------------------------------

def _selective_scan_sequential(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation using an explicit time loop.

    This is the slow-but-obviously-correct version used as ground truth
    in the numerical equivalence tests.
    """
    B_size, L, D_size = x.shape
    N = A.shape[0]
    dtype = x.dtype
    device = x.device

    h = torch.zeros(B_size, D_size, N, dtype=dtype, device=device)
    outs = []

    for t in range(L):
        delta_t = delta[:, t, :]          # (B, D)
        A_t = A.to(dtype=dtype)           # (N,)
        B_t = B[:, t, :]                  # (B, N)
        C_t = C[:, t, :]                  # (B, N)
        x_t = x[:, t, :]                  # (B, D)

        # ZOH discretisation
        A_bar = torch.exp(
            delta_t.unsqueeze(-1) * A_t.view(1, 1, N)
        )  # (B, D, N)
        dBx = (
            delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
        )  # (B, D, N)

        h = A_bar * h + dBx              # (B, D, N)
        y_t = (C_t.unsqueeze(1) * h).sum(-1)  # (B, D)
        if D is not None:
            y_t = y_t + D.to(dtype=dtype, device=device) * x_t
        outs.append(y_t)

    return torch.stack(outs, dim=1)      # (B, L, D)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

try:
    from ternair.model.ssm import _selective_scan_parallel
    HAS_SSM = True
except ImportError:
    HAS_SSM = False


@pytest.mark.skipif(not HAS_SSM, reason="ternair.model.ssm not importable")
class TestParallelScanEquivalence:
    """Numerical equivalence between sequential and parallel scans."""

    @pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    def device(self, request):
        return request.param

    def _make_inputs(
        self,
        B: int = 2,
        L: int = 32,
        D: int = 16,
        N: int = 8,
        device: str = "cpu",
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        x = torch.randn(B, L, D, device=device)
        delta = F.softplus(torch.randn(B, L, D, device=device)) * 0.1
        A = -torch.exp(torch.arange(1, N + 1, dtype=torch.float32, device=device))
        Bm = torch.randn(B, L, N, device=device)
        C = torch.randn(B, L, N, device=device)
        D_vec = torch.ones(D, device=device)
        return x, delta, A, Bm, C, D_vec

    def test_output_shape(self, device):
        """Output shape matches (B, L, D)."""
        x, delta, A, Bm, C, D_vec = self._make_inputs(device=device)
        out = _selective_scan_parallel(x, delta, A, Bm, C, D_vec)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_numerically_close_to_sequential(self, device):
        """Parallel scan matches sequential to within float32 tolerance."""
        x, delta, A, Bm, C, D_vec = self._make_inputs(B=2, L=16, D=8, N=4, device=device)

        out_seq = _selective_scan_sequential(x, delta, A, Bm, C, D_vec)
        out_par = _selective_scan_parallel(x, delta, A, Bm, C, D_vec)

        max_diff = (out_seq - out_par).abs().max().item()
        rel_diff = max_diff / (out_seq.abs().max().item() + 1e-8)

        # Float32 cumsum accumulates numerical error — tolerate up to 1e-4 relative
        assert rel_diff < 1e-4, (
            f"Relative difference between sequential and parallel scan: {rel_diff:.2e} "
            f"(max abs diff: {max_diff:.2e})"
        )

    def test_without_skip_connection(self, device):
        """Parallel scan with D=None."""
        x, delta, A, Bm, C, _ = self._make_inputs(device=device)
        out_seq = _selective_scan_sequential(x, delta, A, Bm, C, None)
        out_par = _selective_scan_parallel(x, delta, A, Bm, C, None)
        max_diff = (out_seq - out_par).abs().max().item()
        assert max_diff < 1e-3, f"max diff without D: {max_diff:.2e}"

    def test_long_sequence(self, device):
        """Parallel scan handles long sequences (L=512)."""
        x, delta, A, Bm, C, D_vec = self._make_inputs(B=1, L=512, D=8, N=4, device=device)
        out_seq = _selective_scan_sequential(x, delta, A, Bm, C, D_vec)
        out_par = _selective_scan_parallel(x, delta, A, Bm, C, D_vec)
        rel_diff = (out_seq - out_par).abs().max().item() / (out_seq.abs().max().item() + 1e-8)
        # Longer sequences accumulate more float32 error — tolerate 5e-4
        assert rel_diff < 5e-4, f"Long-sequence relative diff: {rel_diff:.2e}"

    def test_batch_invariance(self, device):
        """Each item in the batch produces the same output as running alone."""
        x, delta, A, Bm, C, D_vec = self._make_inputs(B=4, L=16, D=8, N=4, device=device)
        out_batch = _selective_scan_parallel(x, delta, A, Bm, C, D_vec)
        for i in range(x.shape[0]):
            out_i = _selective_scan_parallel(
                x[i : i + 1], delta[i : i + 1], A,
                Bm[i : i + 1], C[i : i + 1], D_vec,
            )
            diff = (out_batch[i] - out_i[0]).abs().max().item()
            assert diff < 1e-5, f"Batch item {i} differs: {diff:.2e}"

    def test_stable_with_large_delta(self, device):
        """Scan remains numerically stable with large delta values."""
        torch.manual_seed(0)
        B, L, D, N = 1, 32, 8, 4
        x = torch.randn(B, L, D, device=device)
        # Large delta — tests log-space stability
        delta = torch.ones(B, L, D, device=device) * 5.0
        A = -torch.exp(torch.arange(1, N + 1, dtype=torch.float32, device=device))
        Bm = torch.randn(B, L, N, device=device)
        C = torch.randn(B, L, N, device=device)
        D_vec = torch.ones(D, device=device)

        out = _selective_scan_parallel(x, delta, A, Bm, C, D_vec)
        assert not out.isnan().any(), "NaN in output with large delta"
        assert not out.isinf().any(), "Inf in output with large delta"


@pytest.mark.skipif(not HAS_SSM, reason="ternair.model.ssm not importable")
class TestSSMBlockIntegration:
    """Integration tests for TernarySSMBlock using the parallel scan."""

    def test_ssm_block_forward(self):
        """TernarySSMBlock forward pass produces the right shape."""
        try:
            from ternair.model.ssm import TernarySSMBlock
            from ternair.model.size_profiles import tiny_profile
        except ImportError:
            pytest.skip("ternair.model not importable")

        cfg = tiny_profile()
        block = TernarySSMBlock(cfg)
        block.eval()

        B, L, H = 2, 16, cfg.hidden_size
        x = torch.randn(B, L, H)
        with torch.no_grad():
            out = block(x)

        assert out.shape == (B, L, H), f"Expected ({B}, {L}, {H}), got {out.shape}"
        assert not out.isnan().any(), "NaN in SSM block output"

    def test_ssm_block_gradient_flows(self):
        """Gradients flow through the SSM block (training mode)."""
        try:
            from ternair.model.ssm import TernarySSMBlock
            from ternair.model.size_profiles import tiny_profile
        except ImportError:
            pytest.skip("ternair.model not importable")

        cfg = tiny_profile()
        block = TernarySSMBlock(cfg)
        block.train()

        x = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient on input"
        assert not x.grad.isnan().any(), "NaN gradient"
