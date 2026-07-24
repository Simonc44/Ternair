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
    # rope scaling could be added here if we want to extend prototypes
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Valeurs par defaut conditionnelles
        if self.num_attn_layers < 0:
            if self.attn_layer_period > 0:
                # Pattern periodique: [SSM x (period-1), Attn] x N
                # Nombre de couches d'attention = ceil(num_layers / period)
                self.num_attn_layers = (self.num_hidden_layers + self.attn_layer_period - 1) // self.attn_layer_period
            else:
                self.num_attn_layers = self.num_hidden_layers
        if self.thalamus_dim < 0:
            self.thalamus_dim = self.hidden_size

        # Validations
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be a multiple of num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.storage not in ("int8", "packed", "fastpacked"):
            raise ValueError(f"Unsupported storage mode {self.storage!r}")
        if self.attn_layer_period < 1:
            raise ValueError(f"attn_layer_period must be >= 1, got {self.attn_layer_period}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["TernairConfig"]
