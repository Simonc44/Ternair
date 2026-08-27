"""Model configuration dataclass.

Loosely inspired by :class:`transformers.PretrainedConfig`, but kept
as a plain dataclass so the package remains self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TernairConfig:
    """Hyper-parameters for a ternary decoder-only model."""

    vocab_size: int = 32000
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    # MLP activation: "silu" (SwiGLU, default) or "relu2" (SquaredReLU,
    # used by the official BitNet b1.58 2B-4T checkpoint).
    hidden_act: str = "silu"
    storage: str = "packed"  # one of: "int8", "packed", "fastpacked"
    # Hybrid architecture (SSM + attention)
    num_attn_layers: int = -1  # -1 → all layers are attention (legacy mode)
    attn_layer_period: int = 4  # Pattern: SSM x (period-1) + Attention (ex: 4 = SSM-SSM-SSM-Attn)
    ssm_dim: int = 16
    ssm_dt_rank: str | int = "auto"
    # Thalamic bottleneck
    thalamus_k: int = 32
    thalamus_heads: int = 4
    thalamus_dim: int = -1  # -1 → same as hidden_size
    # MoE configuration
    num_experts: int = 1  # 1 = desactive (pas de MoE)
    top_k_experts: int = 1
    moe_layer_period: int = 0  # 0 = desactive
    # KV-Cache quantifie (BitAttention)
    kv_cache_bits: int = 0  # 0 = pas de quant KV, 2 = 2-bit, 4 = 4-bit
    # BitNet b1.58 sub-layer normalisation (official architecture):
    #   attn: out = attn_sub_norm(attn_out) before o_proj
    #   mlp : out = ffn_sub_norm(silu(gate(x)) * up(x)) before down_proj
    use_sub_norm: bool = False
    # rope scaling could be added here if we want to extend prototypes
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Valeurs par defaut conditionnelles
        if self.num_attn_layers < 0:
            self.num_attn_layers = self.num_hidden_layers
        if self.thalamus_dim < 0:
            self.thalamus_dim = self.hidden_size

        # Validations (renforcees pour v0.5.0)
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be > 0, got {self.intermediate_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be > 0, got {self.num_hidden_layers}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be > 0, got {self.num_attention_heads}")
        if self.num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be > 0, got {self.num_key_value_heads}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be > 0, got {self.max_position_embeddings}"
            )

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be a multiple of "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot "
                f"exceed num_attention_heads ({self.num_attention_heads})"
            )

        if self.storage not in ("int8", "packed", "fastpacked"):
            raise ValueError(f"Unsupported storage mode {self.storage!r}")
        if self.attn_layer_period < 1:
            raise ValueError(
                f"attn_layer_period must be >= 1, got {self.attn_layer_period}"
            )

        # MoE sanity
        if self.num_experts > 1 and self.top_k_experts > self.num_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) cannot exceed "
                f"num_experts ({self.num_experts})"
            )
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")

        # KV-Cache bits sanity
        if self.kv_cache_bits not in (0, 2, 4, 8):
            raise ValueError(
                f"kv_cache_bits must be one of {{0, 2, 4, 8}}, "
                f"got {self.kv_cache_bits}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["TernairConfig"]
