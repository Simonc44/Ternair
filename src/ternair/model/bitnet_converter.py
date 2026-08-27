"""Convert a trained BitNet b1.58 HuggingFace checkpoint into Ternair format.

BitNet b1.58 (e.g. ``microsoft/bitnet-b1.58-2B-4T``) and Ternair share the
same architecture -- LLaMA-style decoder with RMSNorm, RoPE, GQA attention,
SwiGLU MLP, and per-row absmean ternary weights ``gamma = mean(|W|)``.

The official BitNet checkpoints store **master bf16 weights**; inference
ternarises them on the fly.  This module does exactly that once, at
conversion time:

1. Load the HF ``config.json`` + ``model.safetensors`` (master weights).
2. Build a matching :class:`TernairConfig`.
3. Copy every master weight into the matching :class:`TernairLinear`.
4. Call :meth:`TernairForCausalLM.freeze_storage` -- ternarise + pack.
5. Write a native Ternair package (``config.json`` + ``model.safetensors``).

The result is a **fully trained model** (same weights BitNet ships) in
Ternair's denser packing (1.6 bits/value ``packed`` or 2 bits ``fastpacked``),
loadable with :func:`load_converted_model` or the server / CLI.

Usage
-----
.. code-block:: bash

    python -m ternair import-bitnet --source ./bitnet-2b4t --output ./ternair-2b4t --storage packed

.. code-block:: python

    from ternair.model.bitnet_converter import convert_bitnet_checkpoint

    report = convert_bitnet_checkpoint("./bitnet-2b4t", "./ternair-2b4t")
    model, tokenizer = load_converted_model("./ternair-2b4t")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ternair.errors import ArtifactError, ConfigurationError
from ternair.model.config import TernairConfig
from ternair.model.modeling import TernairForCausalLM
from ternair.model.export import export_huggingface_package

_LOGGER = logging.getLogger(__name__)

# HF key suffixes -> Ternair module attribute path (relative to a layer block).
_LAYER_KEY_MAP = {
    "input_layernorm.weight": "ln_1.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    # BitNet b1.58 sub-layer norms (official architecture).
    "self_attn.attn_sub_norm.weight": "attn.attn_sub_norm.weight",
    "mlp.ffn_sub_norm.weight": "mlp.ffn_sub_norm.weight",
    "post_attention_layernorm.weight": "ln_2.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}

_ROOT_KEY_MAP = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}


@dataclass
class ConvertReport:
    """Diagnostic summary of a BitNet -> Ternair conversion."""

    source: str = ""
    output_dir: str = ""
    storage: str = "packed"
    n_layers: int = 0
    n_ternary_params: int = 0
    n_master_params: int = 0
    n_loaded_tensors: int = 0
    n_ignored_tensors: int = 0
    ignored_keys: list[str] = field(default_factory=list)
    size_mib: float = 0.0
    fp16_equivalent_mib: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "output_dir": self.output_dir,
            "storage": self.storage,
            "n_layers": self.n_layers,
            "n_ternary_params": self.n_ternary_params,
            "n_master_params": self.n_master_params,
            "n_loaded_tensors": self.n_loaded_tensors,
            "n_ignored_tensors": self.n_ignored_tensors,
            "ignored_keys": self.ignored_keys[:10],
            "size_mib": round(self.size_mib, 2),
            "fp16_equivalent_mib": round(self.fp16_equivalent_mib, 2),
        }


# ---------------------------------------------------------------------------
# Config mapping
# ---------------------------------------------------------------------------


def bitnet_config_to_ternair(hf_config: dict[str, Any], storage: str = "packed") -> TernairConfig:
    """Map a BitNet b1.58 ``config.json`` dict to :class:`TernairConfig`.

    Parameters
    ----------
    hf_config
        The raw ``config.json`` content of a BitNet / LLaMA-style model.
    storage
        Target Ternair storage: ``"packed"`` (1.6 bits/value) or
        ``"fastpacked"`` (2 bits/value).

    Returns
    -------
    TernairConfig
        A config with the same dimensions as the source model, with all
        hybrid features disabled (pure attention, like BitNet).
    """
    if not isinstance(hf_config, dict):
        raise ConfigurationError("BitNet config.json must be a JSON object")

    def _get(*names: str, default: Any = None) -> Any:
        for n in names:
            if n in hf_config and hf_config[n] is not None:
                return hf_config[n]
        return default

    hidden_size = int(_get("hidden_size", "n_embd", default=2048))
    vocab_size = int(_get("vocab_size", default=32000))
    num_hidden_layers = int(_get("num_hidden_layers", "n_layer", default=24))
    num_attention_heads = int(_get("num_attention_heads", "n_head", default=16))
    num_key_value_heads = int(
        _get("num_key_value_heads", "n_kv_head", default=num_attention_heads)
    )
    intermediate_size = int(
        _get("intermediate_size", "n_inner", default=hidden_size * 4)
    )

    cfg = TernairConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=int(_get("max_position_embeddings", "n_positions", default=4096)),
        rope_theta=float(_get("rope_theta", default=10000.0)),
        rms_norm_eps=float(_get("rms_norm_eps", "layer_norm_epsilon", default=1e-5)),
        tie_word_embeddings=bool(_get("tie_word_embeddings", default=True)),
        storage=storage,
        # Pure attention -- BitNet b1.58 has no SSM / MoE layers.
        num_attn_layers=num_hidden_layers,
        attn_layer_period=1,
        num_experts=1,
        moe_layer_period=0,
        kv_cache_bits=int(_get("kv_cache_bits", default=0)),
        use_sub_norm=bool(_get("use_sub_norm", default=False)),
        extra={
            "source_model_type": _get("model_type", default="bitnet"),
            "source_architectures": _get("architectures", default=[]),
        },
    )
    return cfg


def _ternair_state_dict_path(hf_key: str) -> str | None:
    """Map an HF state-dict key to the corresponding Ternair model path.

    Returns ``None`` for keys that do not correspond to a Ternair tensor
    (e.g. rotary inv_freq buffers, generation config keys).
    """
    if hf_key in _ROOT_KEY_MAP:
        return _ROOT_KEY_MAP[hf_key]
    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".")
        # model.layers.<i>.<rest>
        if len(parts) < 4 or parts[1] != "layers":
            return None
        try:
            layer_idx = int(parts[2])
        except ValueError:
            return None
        rest = ".".join(parts[3:])
        if rest in _LAYER_KEY_MAP:
            return f"model.layers.{layer_idx}.block.{_LAYER_KEY_MAP[rest]}"
    return None


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_safetensors_dict(path: str) -> dict[str, torch.Tensor]:
    """Load every tensor from a ``.safetensors`` file into a dict."""
    from safetensors import safe_open  # type: ignore

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def _set_param(model: TernairForCausalLM, dotted: str, value: torch.Tensor) -> bool:
    """Assign ``value`` to ``model.<dotted>`` if the module tree allows it."""
    parts = dotted.split(".")
    cur: Any = model
    for p in parts[:-1]:
        if not hasattr(cur, p):
            return False
        cur = getattr(cur, p)
    if not hasattr(cur, parts[-1]):
        return False
    target = getattr(cur, parts[-1])
    try:
        with torch.no_grad():
            target.data.copy_(value)
        return True
    except Exception:
        return False


def _find_checkpoint_files(source_dir: str) -> list[str]:
    """Locate all checkpoint shards in ``source_dir``.

    Supports single ``model.safetensors``, multi-shard
    ``model-*.safetensors``, and legacy ``pytorch_model.bin``.
    """
    single = os.path.join(source_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    shards = sorted(
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith(".safetensors") and "index" not in f
    )
    if shards:
        return shards
    legacy = os.path.join(source_dir, "pytorch_model.bin")
    if os.path.exists(legacy):
        return [legacy]
    raise ArtifactError(
        f"No model.safetensors found in {source_dir!r}; "
        "download the checkpoint first (e.g. microsoft/bitnet-b1.58-2B-4T)."
    )


@torch.no_grad()
def convert_bitnet_checkpoint(
    source_dir: str,
    output_dir: str,
    storage: str = "packed",
    copy_tokenizer: bool = True,
) -> ConvertReport:
    """Convert a trained BitNet b1.58 HF checkpoint into a Ternair package.

    Parameters
    ----------
    source_dir
        Directory containing the BitNet ``config.json`` + ``model.safetensors``.
    output_dir
        Where to write the Ternair package (``config.json`` + ``model.safetensors``).
    storage
        ``"packed"`` (default, 1.6 bits/value) or ``"fastpacked"`` (2 bits/value).
    copy_tokenizer
        Copy ``tokenizer.*`` / ``vocab.json`` / ``merges.txt`` files to the
        output so the model can be used with its original tokenizer.

    Returns
    -------
    ConvertReport
        Summary of the conversion (sizes, tensor counts, ignored keys).

    Raises
    ------
    ArtifactError
        If the source checkpoint or required tensors are missing.
    ConfigurationError
        If the source config cannot be mapped to Ternair dimensions.
    """
    if storage not in ("packed", "fastpacked"):
        raise ConfigurationError(f"Unsupported storage {storage!r}; use 'packed' or 'fastpacked'")

    source_dir = os.path.abspath(source_dir)
    output_dir = os.path.abspath(output_dir)
    report = ConvertReport(source=source_dir, output_dir=output_dir, storage=storage)

    config_path = os.path.join(source_dir, "config.json")
    if not os.path.exists(config_path):
        raise ArtifactError(f"Missing {config_path}")

    with open(config_path) as f:
        hf_config = json.load(f)

    ternair_config = bitnet_config_to_ternair(hf_config, storage=storage)
    model = TernairForCausalLM(ternair_config)

    checkpoint_files = _find_checkpoint_files(source_dir)
    _LOGGER.info("Loading master weights from %d shard(s) ...", len(checkpoint_files))
    tensors: dict[str, torch.Tensor] = {}
    for path in checkpoint_files:
        if path.endswith(".bin"):
            tensors.update(torch.load(path, map_location="cpu", weights_only=True))
        else:
            tensors.update(_load_safetensors_dict(path))

    # Detect BitNet b1.58 sub-layer norms from the checkpoint keys and
    # enable them in the config if present (official 2B-4T architecture).
    has_attn_sub_norm = any(k.endswith("self_attn.attn_sub_norm.weight") for k in tensors)
    has_ffn_sub_norm = any(k.endswith("mlp.ffn_sub_norm.weight") for k in tensors)
    if has_attn_sub_norm or has_ffn_sub_norm:
        if not ternair_config.use_sub_norm:
            _LOGGER.info(
                "Checkpoint has sub-layer norms (attn=%s ffn=%s); enabling use_sub_norm",
                has_attn_sub_norm, has_ffn_sub_norm,
            )
            ternair_config.use_sub_norm = True
            model = TernairForCausalLM(ternair_config)

    # Copy master weights into the Ternair model (embedding, norms, linears).
    n_loaded = 0
    for hf_key, tensor in tensors.items():
        tern_key = _ternair_state_dict_path(hf_key)
        if tern_key is None:
            report.n_ignored_tensors += 1
            if len(report.ignored_keys) < 10:
                report.ignored_keys.append(hf_key)
            continue
        if _set_param(model, tern_key, tensor):
            n_loaded += 1
        else:
            report.n_ignored_tensors += 1
            if len(report.ignored_keys) < 10:
                report.ignored_keys.append(hf_key)
    report.n_loaded_tensors = n_loaded

    # Ternarise + pack every linear layer (gamma = mean(|W|) per row).
    snapshot = model.freeze_storage()
    report.n_layers = len(snapshot)
    report.n_ternary_params = model.count_parameters(include_embedding=True)
    report.n_master_params = sum(p.numel() for p in model.parameters())

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    export_huggingface_package(model, output_dir, model_name="Ternair (BitNet b1.58 converted)")

    if copy_tokenizer:
        _copy_tokenizer_files(source_dir, output_dir)

    size_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".safetensors") or f == "config.json"
    )
    report.size_mib = size_bytes / 1024**2
    report.fp16_equivalent_mib = report.n_master_params * 2 / 1024**2

    _LOGGER.info(
        "Converted %d tensors (%d layers, %d ternary params) -> %s",
        n_loaded, report.n_layers, report.n_ternary_params, output_dir,
    )
    return report


def _copy_tokenizer_files(source_dir: str, output_dir: str) -> None:
    """Copy tokenizer-related files so the model keeps its original tokenizer."""
    suffixes = (
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "vocab.json", "merges.txt", "special_tokens_map.json",
        "tokenizer.model.vocab", "added_tokens.json",
    )
    copied = 0
    for name in os.listdir(source_dir):
        if name in suffixes or name.endswith(".tiktoken"):
            src = os.path.join(source_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(output_dir, name))
                copied += 1
    if copied:
        _LOGGER.info("Copied %d tokenizer files", copied)


# ---------------------------------------------------------------------------
# Loading the converted package back
# ---------------------------------------------------------------------------


def load_converted_model(
    model_dir: str,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
):
    """Load a converted Ternair package (native ``packed_weight`` schema).

    Parameters
    ----------
    model_dir
        Output of :func:`convert_bitnet_checkpoint` (or any native Ternair
        export): ``config.json`` + ``model.safetensors``.
    device
        Target device (``"cuda"`` / ``"cpu"``).  ``None`` = auto-detect.
    dtype
        Float dtype for the dequantised weight cache (default FP16).

    Returns
    -------
    (model, tokenizer)
        ``model`` is a :class:`TernairForCausalLM` in eval mode with packed
        storage; ``tokenizer`` is a HuggingFace tokenizer or ``None``.
    """
    from safetensors import safe_open  # type: ignore

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise ArtifactError(f"Missing {config_path}")
    with open(config_path) as f:
        raw = json.load(f)

    cfg = bitnet_config_to_ternair(raw, storage=raw.get("storage", "packed"))
    model = TernairForCausalLM(cfg)

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise ArtifactError(f"Missing {safetensors_path}")

    loaded: dict[str, torch.Tensor] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            loaded[k] = f.get_tensor(k)

    state = model.state_dict()
    missing = [k for k in loaded if k not in state]
    if missing:
        _LOGGER.warning("Ignoring %d unknown tensors: %s", len(missing), missing[:5])

    # Load directly into the model (packed_weight / gamma_eval buffers exist
    # only after freeze_storage; we call it first with the same master-free
    # route: freeze creates empty buffers, then we copy the packed data).
    model.freeze_storage()
    state = model.state_dict()
    for k, v in loaded.items():
        if k not in state:
            continue
        if state[k].shape == v.shape:
            with torch.no_grad():
                state[k].copy_(v)
        else:
            _LOGGER.warning("Shape mismatch for %s: %s vs %s", k, tuple(state[k].shape), tuple(v.shape))

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    tokenizer = None
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as exc:
        _LOGGER.info("Tokenizer not loaded (%s)", exc)

    return model, tokenizer


__all__ = [
    "ConvertReport",
    "bitnet_config_to_ternair",
    "convert_bitnet_checkpoint",
    "load_converted_model",
]
