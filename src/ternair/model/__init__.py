"""Decoder-only ternary transformer (hybrid: attention + SSM blocks)."""

from ternair.model.config import TernairConfig
from ternair.model.modeling import TernairModel, TernairForCausalLM
from ternair.model.attention import TernairAttention
from ternair.model.mlp import TernairMLP
from ternair.model.block import RMSNorm, TernairBlock
from ternair.model.ssm import TernarySSMBlock
from ternair.model.thalamus import ThalamicBottleneck
from ternair.model.hybrid_block import TernairHybridBlock
from ternair.model.generation import generate, generate_stream, format_chat_prompt, decode_tokens
from ternair.model.export import export_to_safetensors, export_huggingface_package, compute_compression_report, print_compression_report
from ternair.model.size_profiles import tiny_profile, base_profile, one_gb_profile
from ternair.model.inference import TernairDirectInferencer, BackendName as DirectBackendName
from ternair.model.loader import (
    InferenceBackend,
    LoadReport,
    TernaryLinearFast,
    load_ternair_model,
    llama_to_hf,
    unpack_2bit,
)

__all__ = [
    "TernairConfig",
    "TernairModel",
    "TernairForCausalLM",
    "TernairAttention",
    "TernairMLP",
    "RMSNorm",
    "TernairBlock",
    "TernarySSMBlock",
    "ThalamicBottleneck",
    "TernairHybridBlock",
    "generate",
    "generate_stream",
    "format_chat_prompt",
    "decode_tokens",
    "TernairDirectInferencer",
    "DirectBackendName",
    "LoadReport",
    "TernaryLinearFast",
    "InferenceBackend",
    "load_ternair_model",
    "llama_to_hf",
    "unpack_2bit",
    "export_to_safetensors",
    "export_huggingface_package",
    "compute_compression_report",
    "print_compression_report",
    "tiny_profile",
    "base_profile",
    "one_gb_profile",
]
