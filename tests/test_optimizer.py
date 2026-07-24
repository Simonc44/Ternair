"""Tests for decoupled weight-decay optimiser."""

from __future__ import annotations

import pytest


def test_create_param_groups_runs_with_ternair_model():
    import torch
    from ternair import TernairForCausalLM, tiny_profile
    from ternair.training.optimizer import create_param_groups

    model = TernairForCausalLM(tiny_profile(storage="packed"))
    groups = create_param_groups(model, lr=1e-3, weight_decay=0.1)
    assert sum(len(g["params"]) for g in groups) > 0

    # Check that ternary weight is in a no-WD group
    all_no_decay_ids = set()
    for g in groups:
        if g["weight_decay"] == 0.0:
            all_no_decay_ids.update(id(p) for p in g["params"])

    from ternair.quantization.linear import TernairLinear
    for name, param in model.named_parameters():
        # Skip buffers that aren't trainable
        if not param.requires_grad:
            continue
        # All TernairLinear.weight params should have weight_decay=0
        # Find the parent module by name
        if "." in name:
            mod_name, pname = name.rsplit(".", 1)
            mod = dict(model.named_modules()).get(mod_name)
            if isinstance(mod, TernairLinear) and pname == "weight":
                assert id(param) in all_no_decay_ids, (
                    f"{name} should be in no-WD group"
                )


def test_create_optimizer_runs():
    import torch
    from ternair import TernairForCausalLM, tiny_profile
    from ternair.training.optimizer import create_optimizer

    model = TernairForCausalLM(tiny_profile())
    optim = create_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert isinstance(optim, torch.optim.AdamW)
    assert len(optim.param_groups) == 3  # ternary, decay, no_decay


def test_clip_gradients_runs():
    import torch
    from ternair import TernairForCausalLM, tiny_profile
    from ternair.training.optimizer import clip_gradients

    model = TernairForCausalLM(tiny_profile())
    ids = torch.randint(0, 4096, (1, 8))
    logits = model(ids)
    logits.sum().backward()
    norm = clip_gradients(model, max_norm=1.0)
    assert norm >= 0.0
    assert isinstance(norm, float)
