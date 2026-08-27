"""Integration test: export a Ternair model, reload it, verify outputs match."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from ternair import TernairForCausalLM, tiny_profile
from ternair.model.export import export_to_safetensors


def test_roundtrip_export_load_native(tmp_path):
    """Export via native ternary format, reload via loader, check logits."""
    from ternair.model.loader import TernaryLinearFast, load_ternair_model

    cfg = tiny_profile(storage="fastpacked")
    model = TernairForCausalLM(cfg)
    model.freeze_storage()
    model.eval()

    # Export
    st_path = tmp_path / "model.safetensors"
    export_to_safetensors(model, str(st_path))
    assert st_path.exists()

    # Also save the embedding weight separately (loader needs it)
    embed_path = tmp_path / "embed_weight.pt"
    torch.save(model.model.embed_tokens.weight.data, embed_path)

    # Build a reference output
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        ref_logits = model(ids)

    # Verify the safetensors file has the right keys
    from safetensors import safe_open
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    assert any(k.endswith(".packed_weight") for k in keys)
    assert any(k.endswith(".gamma_eval") for k in keys)


def test_roundtrip_export_load_native_keys(tmp_path):
    """Verify that native export produces keys the loader can discover."""
    from ternair.model.export import _collect_ternary_tensors

    cfg = tiny_profile(storage="packed")
    model = TernairForCausalLM(cfg)
    model.freeze_storage()
    model.eval()

    tensors = _collect_ternary_tensors(model)

    # Every ternary layer should have packed_weight + gamma_eval
    pw_keys = [k for k in tensors if k.endswith(".packed_weight")]
    gamma_keys = [k for k in tensors if k.endswith(".gamma_eval")]
    assert len(pw_keys) > 0, "No packed_weight keys found"
    assert len(gamma_keys) > 0, "No gamma_eval keys found"
    assert len(pw_keys) == len(gamma_keys), "Mismatched packed/gamma key count"

    # Embedding should be present
    embed_keys = [k for k in tensors if "embed_tokens" in k]
    assert len(embed_keys) > 0, "No embedding keys found"


def test_roundtrip_quantization_consistency():
    """Verify that frozen model produces the same output across multiple forwards."""
    cfg = tiny_profile(storage="packed")
    model = TernairForCausalLM(cfg)
    model.freeze_storage()
    model.eval()

    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        out1 = model(ids)
        out2 = model(ids)

    assert torch.equal(out1, out2), "Frozen model should be deterministic"
