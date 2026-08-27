"""Export Ternair models to HuggingFace-compatible SafeTensors format.

This module provides:

1. :func:`export_to_safetensors` -- Export frozen ternary weights into
   a ``model.safetensors`` file with HuggingFace-compatible metadata.

2. :func:`export_huggingface_package` -- Write a full HuggingFace model
   package (``config.json``, ``model.safetensors``, optional README).

3. :func:`compute_compression_report` -- Compare FP16 vs ternary size.

The exported weights follow the HuggingFace naming convention so they
can be loaded by ``transformers`` if a custom model class is registered.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn

from ternair.errors import ArtifactError
from ternair.model.config import TernairConfig
from ternair.model.modeling import TernairForCausalLM
from ternair.quantization.linear import TernairLinear


# ---------------------------------------------------------------------------
# SafeTensors export helpers
# ---------------------------------------------------------------------------

def _tensor_to_bytes(t: Tensor) -> bytes:
    """Convert a PyTorch tensor to raw bytes (little-endian)."""
    return t.detach().cpu().contiguous().numpy().tobytes()


def _serialize_safetensors(
    tensors: dict[str, Tensor],
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Serialize tensors to the SafeTensors binary format.

    This is a minimal implementation that writes the spec-compliant format
    without depending on the ``safetensors`` package.  If the ``safetensors``
    package is available, use it instead; otherwise this fallback works.

    The SafeTensors format:
      - 8 bytes: header size (uint64, little-endian)
      - N bytes: JSON header (UTF-8)
      - Remaining bytes: tensor data (concatenated)
    """
    header: dict[str, Any] = {}
    offset = 0
    data_chunks: list[bytes] = []

    for name, tensor in tensors.items():
        shape = list(tensor.shape)
        dtype_str = {
            torch.float32: "F32",
            torch.float16: "F16",
            torch.bfloat16: "BF16",
            torch.int8: "I8",
            torch.int16: "I16",
            torch.int32: "I32",
            torch.int64: "I64",
            torch.uint8: "U8",
        }.get(tensor.dtype, "F32")

        byte_size = tensor.numel() * _dtype_bytes(tensor.dtype)
        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + byte_size],
        }
        data_chunks.append(_tensor_to_bytes(tensor))
        offset += byte_size

    if metadata:
        header["__metadata__"] = metadata

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad to 8 bytes as required by spec
    padding = (8 - len(header_bytes) % 8) % 8
    header_bytes = header_bytes + b" " * padding

    header_size = len(header_bytes)
    return (
        header_size.to_bytes(8, "little")
        + header_bytes
        + b"".join(data_chunks)
    )


def _dtype_bytes(dtype: torch.dtype) -> int:
    return {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
    }.get(dtype, 4)


# ---------------------------------------------------------------------------
# Collect tensors from a Ternair model
# ---------------------------------------------------------------------------

def _collect_ternary_tensors(model: TernairForCausalLM) -> dict[str, Tensor]:
    """Collect all tensors from a frozen Ternair model.

    For each TernairLinear, extracts the packed_weight and gamma_eval
    buffers.  Also collects the embedding weight and norm parameters.

    Names follow HuggingFace conventions (e.g. ``model.layers.0.mlp.gate_proj.gamma``).
    """
    tensors: dict[str, Tensor] = {}
    prefix = "model."

    # Embedding
    tensors[f"{prefix}embed_tokens.weight"] = model.model.embed_tokens.weight.data

    # Per-layer norms + TernairLinear weights/gamma
    for name, module in model.named_modules():
        if isinstance(module, TernairLinear):
            if not module.is_frozen():
                raise RuntimeError(
                    f"Module {name} is not frozen. Call model.freeze_storage() first."
                )

            # HuggingFace-style name: replace dots with proper path
            hf_name = name.replace("model.", prefix)

            # Packed weight (uint8)
            if module.packed_weight.numel() > 0:
                tensors[f"{hf_name}.packed_weight"] = module.packed_weight

            # Gamma scale (FP32)
            tensors[f"{hf_name}.gamma_eval"] = module.gamma_eval

            # If alpha is learned, export it too
            if module._use_learned_alpha and module.alpha is not None:
                tensors[f"{hf_name}.alpha"] = module.alpha.data

        elif isinstance(module, nn.Embedding) and name == "model.embed_tokens":
            pass  # Already handled above

    # RMSNorm weights
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) or (
            hasattr(module, "weight") and "RMSNorm" in type(module).__name__
        ):
            hf_name = name.replace("model.", prefix)
            tensors[f"{hf_name}.weight"] = module.weight.data

    # LM head
    if model.lm_head is not None:
        if isinstance(model.lm_head, TernairLinear):
            tensors["lm_head.packed_weight"] = model.lm_head.packed_weight
            tensors["lm_head.gamma"] = model.lm_head.gamma_eval
        else:
            tensors["lm_head.weight"] = model.lm_head.weight.data

    return tensors


# ---------------------------------------------------------------------------
# HuggingFace-compatible config.json
# ---------------------------------------------------------------------------

def build_hf_config(ternair_config: TernairConfig) -> dict:
    """Build a HuggingFace-compatible ``config.json`` dict.

    This uses the ``TernairConfig`` fields mapped to HF-style names.
    The ``model_type`` is set to ``"ternair"`` so that a custom HF
    model class can register itself.
    """
    return {
        "model_type": "ternair",
        "vocab_size": ternair_config.vocab_size,
        "hidden_size": ternair_config.hidden_size,
        "intermediate_size": ternair_config.intermediate_size,
        "num_hidden_layers": ternair_config.num_hidden_layers,
        "num_attention_heads": ternair_config.num_attention_heads,
        "num_key_value_heads": ternair_config.num_key_value_heads,
        "max_position_embeddings": ternair_config.max_position_embeddings,
        "rope_theta": ternair_config.rope_theta,
        "rms_norm_eps": ternair_config.rms_norm_eps,
        "tie_word_embeddings": ternair_config.tie_word_embeddings,
        "storage": ternair_config.storage,
        "attn_layer_period": ternair_config.attn_layer_period,
        "ssm_dim": ternair_config.ssm_dim,
        "thalamus_k": getattr(ternair_config, "thalamus_k", 32),
        "thalamus_heads": getattr(ternair_config, "thalamus_heads", 4),
        "torch_dtype": "float16",
        "architectures": ["TernairForCausalLM"],
        "quantization_config": {
            "quantization_method": "bitnet_b1_58",
            "storage": ternair_config.storage,
            "bits_per_value": 1.6 if ternair_config.storage == "packed"
                            else 2.0 if ternair_config.storage == "fastpacked"
                            else 8.0,
            "use_learned_alpha": True,
            "hadamard_activation": True,
            "swiglu_mlp": True,
            "hybrid_ssm_ratio": f"1:{ternair_config.attn_layer_period - 1}",
        },
    }


def build_metadata(ternair_config: TernairConfig) -> dict[str, str]:
    """Build metadata dict for the safetensors file header.

    Includes the format version so future loaders can validate.
    """
    return {
        "format": "ternair_v1",
        "storage": ternair_config.storage,
        "model_type": "TernairForCausalLM",
        "compression": "bitnet_b1.58",
        "num_params": str(
            ternair_config.vocab_size * ternair_config.hidden_size
            + sum(  # crude estimate
                ternair_config.hidden_size * ternair_config.hidden_size * 2
                + ternair_config.hidden_size * ternair_config.intermediate_size * 3
                for _ in range(ternair_config.num_hidden_layers)
            )
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_to_safetensors(
    model: TernairForCausalLM,
    output_path: str = "model.safetensors",
    include_metadata: bool = True,
) -> str:
    """Export a frozen Ternair model to SafeTensors format.

    Parameters
    ----------
    model:
        A :class:`TernairForCausalLM` that has been frozen via
        :meth:`~TernairForCausalLM.freeze_storage`.
    output_path:
        Where to write the ``.safetensors`` file.
    include_metadata:
        Whether to include a descriptive metadata header.

    Returns
    -------
    output_path
        The path the file was written to.
    """
    if model.training:
        raise RuntimeError("Model must be in eval mode before export.")
    if not output_path.lower().endswith(".safetensors"):
        raise ArtifactError("SafeTensors export path must end with '.safetensors'.")

    tensors = _collect_ternary_tensors(model)
    if not tensors:
        raise ArtifactError("Cannot export an empty model artifact.")
    metadata = build_metadata(model.config) if include_metadata else None

    safetensor_bytes = _serialize_safetensors(tensors, metadata=metadata)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(safetensor_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)

    param_count = sum(t.numel() for t in tensors.values())
    total_bytes = len(safetensor_bytes)
    print(
        f"Exported {len(tensors)} tensors ({param_count:,} params, "
        f"{total_bytes / 1024**2:.2f} MiB) -> {output_path}"
    )
    return output_path


def export_huggingface_package(
    model: TernairForCausalLM,
    output_dir: str = "./ternair-hf",
    model_name: str = "Ternair",
    generate_readme: bool = True,
    safe_serialization: bool = True,
) -> str:
    """Export a full HuggingFace-compatible model package.

    Creates the following files in ``output_dir``:

    * ``config.json``  - model configuration
    * ``model.safetensors``  - frozen ternary weights
    * ``model.safetensors.index.json`` - optional shard index
    * ``README.md``  - auto-generated model card

    Parameters
    ----------
    model:
        A frozen Ternair model (call ``model.freeze_storage()`` first).
    output_dir:
        Directory to write the package into.
    model_name:
        Human-readable model name for the README.
    generate_readme:
        Whether to write an auto-generated README.md.

    Returns
    -------
    output_dir
        The path to the created package directory.
    """
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ArtifactError("output_dir must be a non-empty path")
    if model.training:
        model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # 1. config.json
    hf_config = build_hf_config(model.config)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"Wrote {config_path}")

    # 2. model.safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    export_to_safetensors(model, output_path=safetensors_path)

    # 3. README.md
    if generate_readme:
        readme_path = os.path.join(output_dir, "README.md")
        compression = _estimate_compression(model)
        with open(readme_path, "w") as f:
            f.write(_hf_readme_template(model_name, model.config, compression))
        print(f"Wrote {readme_path}")

    return output_dir


# ---------------------------------------------------------------------------
# Compression report
# ---------------------------------------------------------------------------

def _estimate_compression(model: TernairForCausalLM) -> dict:
    """Estimate FP16 vs Ternair size and compression ratio."""
    fp16_bytes = 0
    ternary_bytes = 0

    for module in model.modules():
        if isinstance(module, TernairLinear):
            fp16_bytes += module.in_features * module.out_features * 2  # FP16
            ternary_bytes += module.state_bytes()
        elif isinstance(module, nn.Embedding):
            fp16_bytes += module.weight.numel() * 2

    # Other params (norms, biases)
    for p in model.parameters():
        if not isinstance(p, nn.Parameter):
            continue
        # Check if already counted in TernairLinear
        if any(isinstance(m, TernairLinear) and p is m.weight for m in model.modules() if isinstance(m, TernairLinear)):
            continue
        fp16_bytes += 0  # norms already counted separately

    ratio = fp16_bytes / max(ternary_bytes, 1)
    return {
        "fp16_size_mib": fp16_bytes / 1024**2,
        "ternary_size_mib": ternary_bytes / 1024**2,
        "compression_ratio": round(ratio, 2),
        "savings_percent": round((1 - ternary_bytes / max(fp16_bytes, 1)) * 100, 1),
    }


def compute_compression_report(
    model: TernairForCausalLM,
) -> dict:
    """Compute a detailed compression report table.

    Returns a dictionary with FP16 vs. ternary size comparison
    suitable for logging or rendering.
    """
    stats = _estimate_compression(model)

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = model.count_parameters(include_embedding=True)

    return {
        "total_parameters": total_params,
        "ternary_parameters": ternary_params,
        "fp16_equivalent_mib": stats["fp16_size_mib"],
        "ternary_size_mib": stats["ternary_size_mib"],
        "compression_ratio": stats["compression_ratio"],
        "savings_percent": stats["savings_percent"],
        "storage_mode": model.config.storage,
    }


def print_compression_report(model: TernairForCausalLM) -> None:
    """Pretty-print the compression report to stdout."""
    report = compute_compression_report(model)

    print("=" * 60)
    print("  Ternair Compression Report")
    print("=" * 60)
    print(f"  Total parameters      : {report['total_parameters']:>12,}")
    print(f"  Ternary parameters    : {report['ternary_parameters']:>12,}")
    print(f"  Storage mode          : {report['storage_mode']:>12}")
    print("-" * 60)
    print(f"  FP16 equivalent       : {report['fp16_equivalent_mib']:>10.2f} MiB")
    print(f"  Ternair size          : {report['ternary_size_mib']:>10.2f} MiB")
    print(f"  Compression ratio     : {report['compression_ratio']:>10.2f}x")
    print(f"  Savings               : {report['savings_percent']:>10.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# README template
# ---------------------------------------------------------------------------

def _hf_readme_template(
    model_name: str,
    config: TernairConfig,
    compression: dict,
) -> str:
    """Generate a model card for the HuggingFace Hub."""
    return f"""---
license: apache-2.0
library_name: ternair
tags:
- bitnet
- b1.58
- ternary
- efficient-ai
- quantized
---

# {model_name}

This model was created with **Ternair** - a BitNet b1.58 inference engine
that stores every weight in `{{-1, 0, +1}}` (1.58-bit quantization).

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Ternair Hybrid (SSM 3:1 + GQA Attention) |
| Hidden size | {config.hidden_size} |
| Layers | {config.num_hidden_layers} |
| Attention heads | {config.num_attention_heads} |
| KV heads | {config.num_key_value_heads} |
| Vocab size | {config.vocab_size} |
| Max position | {config.max_position_embeddings} |
| Storage | {config.storage} |
| MLP | SwiGLU (ternary) |
| Activation quant | Hadamard-smooth 8-bit |
| QAT | Learned alpha + annealing |

## Size & Compression

| Metric | Value |
|--------|-------|
| FP16 equivalent | {compression['fp16_size_mib']:.1f} MiB |
| Ternair size | {compression['ternary_size_mib']:.1f} MiB |
| Compression ratio | {compression['compression_ratio']}x |
| Savings | {compression['savings_percent']}% |

## Usage

```python
# Coming soon: transformers integration
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("your-org/{model_name}")
```

For now, use Ternair directly:

```bash
pip install git+https://github.com/Simonc44/Ternair.git
```

## Training

This model was produced by applying Quantization-Aware Training (QAT)
with distillation from a FP16 teacher model.
"""


__all__ = [
    "export_to_safetensors",
    "export_huggingface_package",
    "compute_compression_report",
    "print_compression_report",
    "build_hf_config",
]
