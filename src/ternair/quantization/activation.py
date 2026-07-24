"""8-bit per-token absmax activation quantization (BitNet b1.58).

During training and inference, hidden activations are quantised to
``int8`` using ``γ_a = max(|x|) / 127`` per token (last dimension).
The forward uses STE; the backward treats the quantisation as identity.

Integrates QuaRot/SpinQuant-style Hadamard transform to smooth
activation outliers before quantization, reducing the quantization
error without adding parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Hadamard Transform (QuaRot / SpinQuant)
# ---------------------------------------------------------------------------

def _hadamard_matrix(n: int, device: torch.device = None) -> Tensor:
    """Construit une matrice d'Hadamard normalisee H_n de taille n.
    
    H_n est une matrice orthogonale (H^T H = I) construite recursivement :
        H_1 = [1]
        H_{2n} = [[H_n,  H_n],
                  [H_n, -H_n]] / sqrt(2)
    La normalisation par sqrt(2) a chaque etape garantit l'orthogonalite.
    
    Args:
        n: Taille de la matrice (doit etre une puissance de 2)
        device: Device cible
    
    Returns:
        Matrice d'Hadamard normalisee de forme (n, n)
    """
    if n & (n - 1) != 0:
        # Si n n'est pas une puissance de 2, on prend la puissance de 2
        # superieure et on tronque (approximation pragmatique)
        n_pow2 = 1
        while n_pow2 < n:
            n_pow2 <<= 1
        if n_pow2 != n:
            # Tronquer n'a pas de sens pour Hadamard ; on leve une erreur
            raise ValueError(
                f"La taille {n} doit etre une puissance de 2 pour la "
                f"transformee d'Hadamard exacte. Utilisez apply_hadamard_fast()."
            )
    
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < n:
        h = torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1),
        ], dim=0) / (2.0 ** 0.5)
    return h


def apply_hadamard_transform(x: Tensor, dim: int = -1) -> Tensor:
    """Applique la transformee d'Hadamard rapide (FWHT) sur la dimension
    specifiee pour lisser les activations avant quantification.
    
    La matrice d'Hadamard H est orthogonale (H^T H = I), ce qui preserve
    l'information tout en redistribuant les outliers sur toutes les
    dimensions, rendant la quantification INT8 plus efficace.
    
    Implementation : transformee recursive in-place (Butterfly) O(n log n)
    sans materialiser la matrice complete.
    
    Args:
        x: Tenseur d'entree de forme (..., n, ...) ou n est puissance de 2
        dim: Dimension sur laquelle appliquer la transformee
    
    Returns:
        Tenseur transforme de meme forme
    """
    n = x.shape[dim]
    
    # Verifier que n est une puissance de 2
    if n & (n - 1) != 0:
        raise ValueError(
            f"La dimension {dim} a une taille {n} qui n'est pas une "
            f"puissance de 2. La FWHT necessite des puissances de 2."
        )
    
    # Transposer pour travailler sur la derniere dimension
    x = x.transpose(dim, -1)
    shape = x.shape
    x = x.reshape(-1, n)
    
    # Transformee rapide de Hadamard (butterfly, out-of-place, O(n log n))
    h = 1
    while h < n:
        step = h * 2
        x_out = torch.empty_like(x)
        for i in range(0, n, step):
            u = x[:, i:i + h]
            v = x[:, i + h:i + step]
            x_out[:, i:i + h] = (u + v) / (2.0 ** 0.5)
            x_out[:, i + h:i + step] = (u - v) / (2.0 ** 0.5)
        x = x_out
        h = step
    
    x = x.reshape(shape)
    x = x.transpose(dim, -1)
    return x


def apply_inverse_hadamard(x: Tensor, dim: int = -1) -> Tensor:
    """Applique la transformee inverse (identique car H^T = H)."""
    return apply_hadamard_transform(x, dim=dim)


# ---------------------------------------------------------------------------
# Activation 8-bit Quantisation (BitNet b1.58)
# ---------------------------------------------------------------------------


@dataclass
class Activation8Bit:
    quantised: Tensor  # int8, same shape as input
    scale: Tensor  # fp32, shape broadcastable to (..., 1)


class _ActivationQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, use_hadamard: bool = True) -> Tensor:  # type: ignore[override]
        if use_hadamard:
            x = apply_hadamard_transform(x, dim=-1)
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        scale = absmax / 127.0
        x_int = torch.clamp(torch.round(x / scale), -128.0, 127.0)
        result = x_int.to(torch.float32) * scale
        if use_hadamard:
            result = apply_inverse_hadamard(result, dim=-1)
        return result

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        return grad_out, None


def quantize_activations_8bit(x: Tensor, use_hadamard: bool = True) -> Activation8Bit:
    """Quantise ``x`` per-token to ``int8`` using absmax.

    Si ``use_hadamard`` est True, applique une transformee d'Hadamard
    avant quantification pour lisser les outliers (QuaRot-style).

    Returns both the quantized values (int8) and the per-token
    scale so they can be inspected or recombined outside the STE
    function.
    """
    if use_hadamard:
        x = apply_hadamard_transform(x, dim=-1)
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(x / scale), -128.0, 127.0).to(torch.int8)
    return Activation8Bit(quantised=q, scale=scale.detach().to(torch.float32))


def quantize_activations_8bit_forward(x: Tensor, use_hadamard: bool = True) -> Tensor:
    """Forward pass for activations with STE for backprop.
    
    Si ``use_hadamard`` est True, applique une transformee d'Hadamard
    avant quantification et son inverse apres (QuaRot-style).
    """
    if use_hadamard:
        x = apply_hadamard_transform(x, dim=-1)
        result = _ActivationQuantFn.apply(x, False)
        return apply_inverse_hadamard(result, dim=-1)
    return _ActivationQuantFn.apply(x, False)


__all__ = [
    "Activation8Bit",
    "apply_hadamard_transform",
    "apply_inverse_hadamard",
    "quantize_activations_8bit",
    "quantize_activations_8bit_forward",
]
