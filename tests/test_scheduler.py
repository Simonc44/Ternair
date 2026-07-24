"""Tests for WSD scheduler."""

from __future__ import annotations

import math

import pytest

from ternair.training.scheduler import WSDScheduler


@pytest.fixture
def dummy_opt():
    """A fake optimizer with a single param group."""
    import torch
    return torch.optim.AdamW([torch.nn.Parameter(torch.randn(8))], lr=1.0)


def test_warmup_phase(dummy_opt):
    sched = WSDScheduler(
        dummy_opt, total_steps=100, warmup_steps=10, stable_steps=80, decay_steps=10, min_lr=0.0
    )
    sched.last_epoch = 0
    assert sched.get_lr()[0] == pytest.approx(0.0, abs=1e-6)
    sched.last_epoch = 5
    assert sched.get_lr()[0] == pytest.approx(0.5, abs=1e-6)
    sched.last_epoch = 10
    assert sched.get_lr()[0] == pytest.approx(1.0, abs=1e-6)


def test_stable_phase(dummy_opt):
    sched = WSDScheduler(
        dummy_opt, total_steps=100, warmup_steps=10, stable_steps=80, decay_steps=10, min_lr=0.0
    )
    sched.last_epoch = 50
    assert sched.get_lr()[0] == pytest.approx(1.0, abs=1e-6)
    sched.last_epoch = 89
    assert sched.get_lr()[0] == pytest.approx(1.0, abs=1e-6)


def test_cosine_decay_phase(dummy_opt):
    sched = WSDScheduler(
        dummy_opt, total_steps=100, warmup_steps=10, stable_steps=80, decay_steps=10, min_lr=0.0
    )
    # Decay starts at step 90
    sched.last_epoch = 90
    assert sched.get_lr()[0] == pytest.approx(0.5 * (1.0 + math.cos(0.0)), abs=1e-6)
    sched.last_epoch = 95
    mid_factor = 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert sched.get_lr()[0] == pytest.approx(mid_factor, abs=1e-6)
    sched.last_epoch = 99  # last decay step
    end_factor = 0.5 * (1.0 + math.cos(math.pi * 0.9))
    assert sched.get_lr()[0] == pytest.approx(end_factor, abs=1e-6)


def test_linear_decay_phase(dummy_opt):
    sched = WSDScheduler(
        dummy_opt, total_steps=100, warmup_steps=10, stable_steps=80, decay_steps=10,
        min_lr=0.2, decay_type="linear",
    )
    sched.last_epoch = 90
    # 1.0 - 0.0 = 1.0 → factor = 1.0 → lr = 0.2 + (1.0 - 0.2) * 1.0 = 1.0
    assert sched.get_lr()[0] == pytest.approx(0.2 + 0.8 * 1.0, abs=1e-6)
    sched.last_epoch = 95
    # progress = 0.5 → factor = 0.5
    assert sched.get_lr()[0] == pytest.approx(0.2 + 0.8 * 0.5, abs=1e-6)
    sched.last_epoch = 99
    # progress = 0.9 → factor = 0.1
    assert sched.get_lr()[0] == pytest.approx(0.2 + 0.8 * 0.1, abs=1e-6)


def test_past_end_goes_to_min_lr(dummy_opt):
    sched = WSDScheduler(
        dummy_opt, total_steps=100, warmup_steps=10, stable_steps=80, decay_steps=10, min_lr=0.0
    )
    sched.last_epoch = 200
    assert sched.get_lr()[0] == pytest.approx(0.0, abs=1e-6)
