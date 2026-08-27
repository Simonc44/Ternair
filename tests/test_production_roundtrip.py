"""Production-oriented regression tests for export and inference."""

from __future__ import annotations

import json

import pytest
import torch

from ternair import TernairForCausalLM, tiny_profile
from ternair.model.export import export_to_safetensors


def test_export_writes_canonical_native_keys_and_is_atomic(tmp_path):
    model = TernairForCausalLM(tiny_profile(storage="fastpacked"))
    model.freeze_storage()
    model.eval()
    path = tmp_path / "model.safetensors"
    export_to_safetensors(model, str(path))

    assert path.exists()
    assert not (tmp_path / "model.safetensors.tmp").exists()
    raw = path.read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    tensor_keys = set(header) - {"__metadata__"}
    assert any(key.endswith(".packed_weight") for key in tensor_keys)
    assert any(key.endswith(".gamma_eval") for key in tensor_keys)


def test_generation_resets_cache_between_requests():
    model = TernairForCausalLM(tiny_profile(storage="fastpacked"))
    model.freeze_storage()
    model.eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    from ternair.model.generation import generate

    first = generate(model, prompt, max_new_tokens=2, temperature=0.0)
    second = generate(model, prompt, max_new_tokens=2, temperature=0.0)
    assert torch.equal(first, second)


def test_invalid_configuration_has_public_exception():
    from ternair import ConfigurationError

    with pytest.raises(ValueError):
        tiny_profile(hidden_size=0)
    assert issubclass(ConfigurationError, ValueError)
