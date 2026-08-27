"""Round-trip test: synthetic BitNet b1.58 checkpoint -> Ternair package.

Builds a tiny LLaMA-style checkpoint with HF key names and bf16 master
weights, converts it with :func:`convert_bitnet_checkpoint`, then reloads
it with :func:`load_converted_model` and verifies that:

* all expected tensors are loaded (no silent key loss),
* the frozen model runs end-to-end and produces the same logits as a
  model built directly from the same master weights (fidelity),
* the exported package is smaller than the FP16 equivalent.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # type: ignore

from ternair.model.bitnet_converter import (
    convert_bitnet_checkpoint,
    load_converted_model,
    bitnet_config_to_ternair,
)


def _make_synthetic_checkpoint(tmp_path, hidden=32, layers=2, heads=4, kv=2):
    """Write a tiny BitNet-style checkpoint (config.json + model.safetensors)."""
    vocab = 64
    intermediate = hidden * 2
    cfg = {
        "model_type": "bitnet",
        "architectures": ["BitnetForCausalLM"],
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    tensors: dict[str, np.ndarray] = {}
    tensors["model.embed_tokens.weight"] = np.random.randn(vocab, hidden).astype(np.float16)
    tensors["model.norm.weight"] = np.random.randn(hidden).astype(np.float16)
    for i in range(layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.random.randn(hidden).astype(np.float16)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.random.randn(hidden).astype(np.float16)
        tensors[f"{p}.self_attn.q_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float16)
        tensors[f"{p}.self_attn.k_proj.weight"] = np.random.randn(hidden // 2, hidden).astype(np.float16)
        tensors[f"{p}.self_attn.v_proj.weight"] = np.random.randn(hidden // 2, hidden).astype(np.float16)
        tensors[f"{p}.self_attn.o_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float16)
        tensors[f"{p}.mlp.gate_proj.weight"] = np.random.randn(intermediate, hidden).astype(np.float16)
        tensors[f"{p}.mlp.up_proj.weight"] = np.random.randn(intermediate, hidden).astype(np.float16)
        tensors[f"{p}.mlp.down_proj.weight"] = np.random.randn(hidden, intermediate).astype(np.float16)

    torch_tensors = {
        k: torch.from_numpy(v).to(torch.bfloat16) for k, v in tensors.items()
    }
    save_file(torch_tensors, str(tmp_path / "model.safetensors"))
    return cfg


def _frozen_logits_from_master(ternair_cfg, master_path: str, ids) -> torch.Tensor:
    """Reference: build a Ternair model, load master weights, freeze, forward."""
    from ternair.model.modeling import TernairForCausalLM
    from ternair.model.bitnet_converter import _load_safetensors_dict, _set_param, _ternair_state_dict_path

    model = TernairForCausalLM(ternair_cfg)
    for k, v in _load_safetensors_dict(master_path).items():
        tk = _ternair_state_dict_path(k)
        if tk is not None:
            _set_param(model, tk, v)
    model.freeze_storage()
    model.eval()
    with torch.no_grad():
        return model(ids)


def test_config_mapping_roundtrip():
    hf = {
        "vocab_size": 32000, "hidden_size": 2048, "intermediate_size": 8192,
        "num_hidden_layers": 24, "num_attention_heads": 16,
        "num_key_value_heads": 8, "max_position_embeddings": 4096,
        "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    cfg = bitnet_config_to_ternair(hf, storage="packed")
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 8
    assert cfg.tie_word_embeddings is False
    # BitNet b1.58 is pure attention: no SSM / MoE.
    assert cfg.num_attn_layers == 24
    assert cfg.attn_layer_period == 1
    assert cfg.num_experts == 1
    assert cfg.storage == "packed"


def test_convert_and_reload_roundtrip(tmp_path):
    src = tmp_path / "bitnet"
    out = tmp_path / "ternair"
    src.mkdir()
    _make_synthetic_checkpoint(src)

    report = convert_bitnet_checkpoint(str(src), str(out), storage="packed")
    d = report.as_dict()

    # Every real tensor must be loaded (embed + norm + 2 layers x 9 weights).
    assert d["n_loaded_tensors"] == 2 + 2 * 9, d
    assert d["n_ignored_tensors"] == 0, d["ignored_keys"]
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    # Packed ternary weights must be much smaller than the FP16 master.
    assert d["size_mib"] < d["fp16_equivalent_mib"]

    # Reload the converted package and check it runs.
    model, tokenizer = load_converted_model(str(out), device="cpu")
    ids = torch.tensor([[1, 5, 20, 3, 42]], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 5, 64)
    assert torch.isfinite(logits).all()


def test_fidelity_against_master_weights(tmp_path):
    """Converted frozen model must match a directly-frozen reference closely."""
    src = tmp_path / "bitnet"
    out = tmp_path / "ternair"
    src.mkdir()
    cfg = _make_synthetic_checkpoint(src)

    report = convert_bitnet_checkpoint(str(src), str(out), storage="packed")
    ternair_cfg = bitnet_config_to_ternair(cfg, storage="packed")

    ids = torch.tensor([[2, 9, 33, 7, 1]], dtype=torch.long)

    ref = _frozen_logits_from_master(ternair_cfg, str(src / "model.safetensors"), ids)
    model, _ = load_converted_model(str(out), device="cpu")
    with torch.no_grad():
        got = model(ids)

    # Same master weights -> same ternary weights -> bit-exact logits
    # (packing is deterministic and lossless for the frozen path).
    max_diff = float((ref - got).abs().max().item())
    assert max_diff < 1e-3, f"frozen logits diverged: max_diff={max_diff}"


def test_parity_with_transformers_bitnet(tmp_path):
    """Converted model must stay close to the real HF BitNetForCausalLM.

    transformers' BitNet keeps master bf16 weights (no on-the-fly
    ternarisation in 5.x), so this is a *quality* check: the ternary
    model must preserve the ranking of the bf16 model, not be bit-equal.
    """
    pytest.importorskip("transformers")
    from transformers.models.bitnet import BitNetConfig, BitNetForCausalLM

    torch.manual_seed(0)
    cfg = BitNetConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-5,
        tie_word_embeddings=True, torch_dtype=torch.bfloat16,
    )
    hf_model = BitNetForCausalLM(cfg).to(torch.bfloat16)
    src = tmp_path / "bitnet-hf"
    src.mkdir()
    hf_model.save_pretrained(str(src))
    out = tmp_path / "ternair-hf"

    report = convert_bitnet_checkpoint(str(src), str(out), storage="packed")
    assert report.as_dict()["n_ignored_tensors"] == 0, report.as_dict()["ignored_keys"]

    ids = torch.tensor([[1, 5, 20, 3, 42]], dtype=torch.long)
    hf_model.eval()
    with torch.no_grad():
        ref = hf_model(ids).logits.float()
    t_model, _ = load_converted_model(str(out), device="cpu")
    with torch.no_grad():
        got = t_model(ids).float()

    # Structure preservation: Pearson correlation between the bf16 and the
    # ternary logits.  On a *random* model the logits are near-uniform so
    # top-1 agreement is meaningless; on a trained checkpoint (the real
    # 2B-4T) we expect correlation ~1.0 and high top-1 agreement.
    rf = ref.flatten().float()
    gf = got.flatten().float()
    rf = rf - rf.mean()
    gf = gf - gf.mean()
    corr = float(((rf * gf).sum() / (rf.norm() * gf.norm() + 1e-8)).item())
    assert corr > 0.5, f"logit correlation too low: {corr}"

    # Sanity: outputs are finite and same-shaped.
    assert ref.shape == got.shape
    assert torch.isfinite(got).all()


def test_fastpacked_storage_conversion(tmp_path):
    src = tmp_path / "bitnet"
    out = tmp_path / "ternair-fast"
    src.mkdir()
    _make_synthetic_checkpoint(src)
    report = convert_bitnet_checkpoint(str(src), str(out), storage="fastpacked")
    assert report.as_dict()["storage"] == "fastpacked"
    model, _ = load_converted_model(str(out), device="cpu")
    ids = torch.tensor([[3, 1, 4, 1, 5]], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
    assert torch.isfinite(logits).all()
