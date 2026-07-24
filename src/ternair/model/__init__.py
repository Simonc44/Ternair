"""Decoder-only ternary transformer (hybrid: attention + SSM blocks)."""

from ternair.model.config import TernairConfig
from ternair.model.modeling import TernairModel, TernairForCausalLM
from ternair.model.attention import TernairAttention
from ternair.model.mlp import TernairMLP
from ternair.model.block import RMSNorm, TernairBlock
from ternair.model.ssm import TernarySSMBlock
from ternair.model.thalamus import ThalamicBottleneck
from ternair.model.hybrid_block import TernairHybridBlock
from ternair.model.generation import generate
from ternair.model.size_profiles import tiny_profile, base_profile, one_gb_profile

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
    "tiny_profile",
    "base_profile",
    "one_gb_profile",
]
