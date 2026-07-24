"""Ternary MoE — Mixture of ternary experts for ultra-sparse inference.

Combines 1.58-bit quantification with a Mixture-of-Experts architecture.
Seule une fraction des experts ternaires (typiquement 2 sur 8) est
activee pour chaque token, reduisant le cout de calcul a ~1/4 d'un
modele dense equivalent.

Architecture
------------
Chaque expert est un petit MLP ternaire (SwiGLU avec TernairLinear).
Un routeur binaire selectionne les top-K experts par token.
Les sorties des experts actifs sont ponderees par les poids de routage.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ternair.model.config import TernairConfig
from ternair.model.mlp import TernairMLP
from ternair.quantization.activation import quantize_activations_8bit_forward
from ternair.quantization.linear import TernairLinear


class TernaryMoEBlock(nn.Module):
    """Bloc MoE avec experts ternaires et routage binaire.

    Args:
        config: Configuration du modele.
        num_experts: Nombre total d'experts.
        top_k: Nombre d'experts actifs par token.
        hidden_size: Dimension cachee.
        intermediate_size: Dimension intermediaire de chaque expert.
    """

    def __init__(
        self,
        config: TernairConfig,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        H = hidden_size or config.hidden_size
        I = intermediate_size or config.intermediate_size

        # Routeur binaire (FP16 — negligeable, <0.1% des parametres)
        self.router = nn.Linear(H, num_experts, bias=False)

        # Experts ternaires (chaque expert est un MLP SwiGLU)
        # On cree des MLP avec la config du modele
        expert_config = TernairConfig(
            hidden_size=H,
            intermediate_size=I,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            storage=config.storage,
        )
        self.experts = nn.ModuleList([
            TernairMLP(expert_config)
            for _ in range(num_experts)
        ])

        # Biais de load-balancing (Z-loss)
        self.register_buffer("expert_bias", torch.zeros(num_experts), persistent=True)

    def _routing(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Calcule les poids de routage pour chaque token.

        Args:
            x: (B, T, H) activations d'entree.

        Returns:
            routing_weights: (B, T, top_k) poids normalises des top-k experts.
            expert_indices: (B, T, top_k) indices des top-k experts selectionnes.
            router_logits: (B, T, num_experts) logits bruts du routeur.
        """
        B, T, H = x.shape
        router_logits = self.router(x)  # (B, T, num_experts)
        router_probs = F.softmax(router_logits + self.expert_bias, dim=-1)

        # Selection top-k
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Re-normalisation des poids des top-k
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return top_k_weights, top_k_indices, router_logits

    def _compute_auxiliary_loss(self, router_logits: Tensor) -> Tensor:
        """Perte de load-balancing pour encourager une utilisation uniforme des experts.

        Utilise la perte Z-loss (ZeRO++ style) + importance loss.
        """
        router_probs = F.softmax(router_logits, dim=-1)
        # Importances = moyenne des probabilites sur le batch
        importance = router_probs.mean(dim=(0, 1))  # (num_experts,)
        # Variance de l'importance -> penaliser le desequilibre
        aux_loss = importance.var() * self.num_experts
        return aux_loss

    def forward(self, x: Tensor) -> Tensor:
        """Forward MoE.

        Args:
            x: (B, T, H) activations d'entree.

        Returns:
            (B, T, H) sortie apres combinaison ponderee des experts.
        """
        B, T, H = x.shape
        residual = x

        # Routing
        x_q = quantize_activations_8bit_forward(x)
        routing_weights, expert_indices, router_logits = self._routing(x_q)

        # Sortie cumulee
        final_output = torch.zeros_like(x)

        # Pour chaque expert, traiter les tokens qui lui sont assignes
        for expert_idx, expert in enumerate(self.experts):
            # Trouver les tokens assignes a cet expert
            mask = (expert_indices == expert_idx).any(dim=-1)  # (B, T)
            if not mask.any():
                continue

            # Extraire les tokens assignes
            expert_input = x[mask]  # (N, H)
            expert_output = expert(expert_input.unsqueeze(0)).squeeze(0)  # (N, H)

            # Poids de routage pour cet expert
            weight_mask = (expert_indices == expert_idx)  # (B, T, top_k)
            # Prendre le poids correspondant
            weights = routing_weights[weight_mask]  # (N,)

            # Ajouter la contribution ponderee
            expert_output = expert_output * weights.unsqueeze(-1)
            final_output[mask] = final_output[mask] + expert_output

        # Perte auxiliaire de load-balancing
        self._aux_loss = self._compute_auxiliary_loss(router_logits)

        return quantize_activations_8bit_forward(final_output)


def add_moe_to_model(
    model: nn.Module,
    num_experts: int = 8,
    top_k: int = 2,
    moe_layer_period: int = 2,
) -> None:
    """Remplace les MLP des couches selectionnees par des blocs MoE.

    Args:
        model: Modele Ternair (TernairForCausalLM).
        num_experts: Nombre d'experts par bloc MoE.
        top_k: Nombre d'experts actifs par token.
        moe_layer_period: Toutes les N couches sont converties en MoE.
    """
    from ternair.model.block import TernairBlock
    from ternair.model.hybrid_block import TernairHybridBlock

    conversions = 0
    for name, module in model.named_modules():
        # Ne convertir que les blocs d'attention (TernairBlock, pas SSM)
        layer_idx = None
        if isinstance(module, TernairBlock):
            layer_idx = getattr(module, "layer_idx", None)
        elif isinstance(module, TernairHybridBlock):
            layer_idx = module.layer_idx

        if layer_idx is not None and layer_idx % moe_layer_period == 0:
            # Remplacer le MLP par un bloc MoE
            config = getattr(model, "config", None)
            if config is not None and hasattr(module, "mlp"):
                moe = TernaryMoEBlock(
                    config,
                    num_experts=num_experts,
                    top_k=top_k,
                )
                module.mlp = moe
                conversions += 1

    print(f"MoE: {conversions} couches converties ({num_experts} experts, top-{top_k})")


__all__ = [
    "TernaryMoEBlock",
    "add_moe_to_model",
]
