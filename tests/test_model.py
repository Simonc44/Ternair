"""Tests for the model: forward, freeze + inference forward, generation, parameter count."""

from __future__ import annotations

import pytest


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_torch(), reason="torch required")


@pytest.fixture
def tiny_model_packed():
    import torch

    from ternair import TernairForCausalLM, tiny_profile

    torch.manual_seed(0)
    cfg = tiny_profile(storage="packed")
    model = TernairForCausalLM(cfg)
    return model, cfg


def test_forward_runs_training_mode(tiny_model_packed):
    import torch

    model, cfg = tiny_model_packed
    model.train()
    ids = torch.randint(0, cfg.vocab_size, size=(2, 8))
    logits = model(ids)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert logits.requires_grad


def test_freeze_then_forward_runs(tiny_model_packed):
    import torch

    model, cfg = tiny_model_packed
    model.freeze_storage()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, size=(2, 8))
    with torch.no_grad():
        out = model(ids)
    assert out.shape == (2, 8, cfg.vocab_size)
    assert not out.requires_grad


def test_greedy_generation_runs(tiny_model_packed):
    import torch

    from ternair import generate

    model, cfg = tiny_model_packed
    model.freeze_storage()
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, size=(1, 4))
    out = generate(model, ids, max_new_tokens=4, eos_token_id=cfg.vocab_size - 1)
    assert out.shape == (1, 8)


def test_count_parameters_includes_embedding(tiny_model_packed):
    model, _ = tiny_model_packed
    total_with = model.count_parameters(include_embedding=True)
    total_without = model.count_parameters(include_embedding=False)
    # The embedding should add something non-zero.
    assert total_with > total_without


def test_num_bytes_positive(tiny_model_packed):
    model, _ = tiny_model_packed
    assert model.num_bytes() > 0


def test_one_gb_profile_size_under_1gb_after_fit():
    from ternair import describe_size, fit_one_gb, one_gb_profile
    from ternair.benchmark.size import model_size_bytes

    cfg = fit_one_gb(one_gb_profile())
    print(describe_size(cfg))
    b = model_size_bytes(cfg)
    # Target ≤ 950 MiB safety budget; accept ≤ 975 MiB for rounding.
    assert b.total_bytes <= int(1024 ** 2 * 975), (
        f"fit_one_gb did not land under 975 MiB: {b.total_bytes / 1024 ** 2:.1f} MiB"
    )
    assert b.total_bytes < int(1024 ** 3), (
        f"Exceeds 1 GiB: {b.total_gib:.3f} GiB"
    )
