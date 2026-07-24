"""Tests for the size estimator (no torch required)."""

from __future__ import annotations

import pytest

from ternair import (
    SizeBreakdown,
    auto_fit_to_bytes,
    base_profile,
    describe_size,
    fit_one_gb,
    model_size_bytes,
    one_gb_profile,
    tiny_profile,
)


def test_size_breakdown_is_dataclass_with_gib_mib() -> None:
    cfg = tiny_profile()
    b = model_size_bytes(cfg)
    assert isinstance(b, SizeBreakdown)
    assert b.total_bytes > 0
    assert b.total_mib > 0
    assert b.total_gib > 0
    # total_gib * 1024**3 should round-trip to total_bytes.
    assert b.total_bytes == int(round(b.total_gib * 1024 ** 3))


def test_tiny_profile_packed_under_10_mib() -> None:
    b = model_size_bytes(tiny_profile())
    # 25 M ternary params at 1.6 b/v = ~5 MB + small embedding/LM head
    assert b.total_mib < 10.0


def test_one_gb_profile_under_1_gib_after_fit() -> None:
    cfg = fit_one_gb(one_gb_profile())
    b = model_size_bytes(cfg)
    # We target ≤ 950 MiB safety budget, but accept ≤ 975 MiB
    # because per-architecture rounding can nudge the solver.
    assert b.total_bytes <= int(1024 ** 2 * 975), (
        f"1 GiB profile > safety budget: {b.total_bytes / 1024 ** 2:.1f} MiB"
    )
    # Hard ceiling: the whole point is staying under 1 GiB
    assert b.total_bytes < int(1024 ** 3), f"Exceeds 1 GiB: {b.total_gib:.3f} GiB"


def test_packed_smaller_than_int8_for_same_config() -> None:
    base = base_profile()
    packed = model_size_bytes(base)
    int8 = model_size_bytes(TernairConfig_replacement(base, storage="int8"))
    assert packed.total_bytes < int8.total_bytes


def TernairConfig_replacement(cfg, **kw):
    # Tiny helper: copy dataclass with overrides.
    import dataclasses

    return dataclasses.replace(cfg, **kw)


def test_auto_fit_returns_finite_layers() -> None:
    cfg = auto_fit_to_bytes(
        base_profile(),
        target_bytes=int(1024 ** 3 * 0.4),  # 400 MiB target
        min_layers=4,
        max_layers=128,
    )
    b = model_size_bytes(cfg)
    assert cfg.num_hidden_layers >= 4
    assert cfg.num_hidden_layers <= 128
    assert b.total_bytes <= int(1024 ** 3 * 0.4) + 50 * 1024 ** 2  # within ~50 MiB


def test_describe_does_not_crash() -> None:
    print(describe_size(one_gb_profile()))
